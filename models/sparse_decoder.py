import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDecoder(nn.Module):
    """Sparse Point Cloud and Pose Transformer Decoder from paper Section 3.3.
    - 10 Transformer decoder layers
    - 8 attention heads
    - token dim: 512
    - queries: 1 learnable pose token + 2048 learnable point tokens (total 2049 tokens)
    - cross-attends to image features (257x1024) + hand feature (1x1024)
    - output heads:
        pose token -> object translation relative to hand palm: to in R^3
        point tokens -> sparse object point cloud: ps_o in R^(2048 x 3)
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_sparse_points: int = 2048,
        image_feat_dim: int = 1024,
        hand_feat_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sparse_points = num_sparse_points

        # 1 Pose query token + Ns point query tokens
        self.pose_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.point_tokens = nn.Parameter(torch.randn(1, num_sparse_points, hidden_dim) * 0.02)

        # Projection from encoder feature dimension (1024) to decoder hidden dim (512)
        self.memory_proj = nn.Linear(image_feat_dim, hidden_dim)

        # 10-layer Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output prediction heads
        # Pose head: predicts hand-relative 3D translation t_o (no rotation)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

        # Point head: predicts 3D coordinates for 2048 points
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

    def forward(
        self,
        fv: torch.Tensor,
        fh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fv: (B, 257, 1024) image feature tokens
            fh: (B, 1024) hand feature vector
        Returns:
            to: (B, 3) hand-relative object translation
            p_s: (B, 2048, 3) sparse object point cloud
        """
        B = fv.shape[0]

        # 1. Combine image tokens and hand feature into memory
        # fv: (B, 257, 1024), fh: (B, 1, 1024) -> (B, 258, 1024)
        fh_token = fh.unsqueeze(1)
        memory_in = torch.cat([fv, fh_token], dim=1)           # (B, 258, 1024)
        memory = self.memory_proj(memory_in)                    # (B, 258, 512)

        # 2. Expand query tokens for batch
        # Pose token: (B, 1, 512), Point tokens: (B, 2048, 512)
        pose_queries = self.pose_token.expand(B, -1, -1)
        point_queries = self.point_tokens.expand(B, -1, -1)
        queries = torch.cat([pose_queries, point_queries], dim=1)  # (B, 2049, 512)

        # 3. Transformer decoding with self-attention & cross-attention
        tgt = queries
        for layer in self.transformer_decoder.layers:
            if self.training and (tgt.requires_grad or memory.requires_grad):
                try:
                    tgt = torch.utils.checkpoint.checkpoint(layer, tgt, memory, use_reentrant=False)
                except TypeError:
                    tgt = torch.utils.checkpoint.checkpoint(layer, tgt, memory)
            else:
                tgt = layer(tgt, memory)
        out = self.norm(tgt)

        # 4. Predict outputs
        pose_feat = out[:, 0, :]                               # (B, 512)
        point_feats = out[:, 1:, :]                            # (B, 2048, 512)

        to = self.pose_head(pose_feat)                         # (B, 3)
        p_s = self.point_head(point_feats)                     # (B, 2048, 3)

        return to, p_s

