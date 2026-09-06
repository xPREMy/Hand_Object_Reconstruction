import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gathers tensor x according to neighbor indices idx.
    Args:
        x: (B, M, C)
        idx: (B, N, K)
    Returns:
        gathered: (B, N, K, C)
    """
    B, N, K = idx.shape
    C = x.shape[-1]
    # Expand indices for gather: (B, N * K, C)
    idx_expanded = idx.view(B, N * K, 1).expand(-1, -1, C)
    gathered = torch.gather(x, 1, idx_expanded)
    return gathered.view(B, N, K, C)


class LocalKNNSelfAttention(nn.Module):
    """Local kNN Self-Attention layer (k=16).
    Aggregates spatial and visual context from the k nearest neighbors.
    """
    def __init__(self, channels: int = 128, k: int = 16):
        super().__init__()
        self.k = k
        self.channels = channels

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels)
        )
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        q_coords: torch.Tensor,
        q_feats: torch.Tensor,
        k_coords: torch.Tensor,
        k_feats: torch.Tensor,
        chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Args:
            q_coords: (B, N_q, 3) query 3D points
            q_feats:  (B, N_q, C) query features
            k_coords: (B, N_k, 3) key/value 3D points
            k_feats:  (B, N_k, C) key/value features
            chunk_size: number of query points to process per chunk to prevent CUDA OOM
        Returns:
            out_feats: (B, N_q, C) context-aggregated features
        """
        B, N_q, _ = q_coords.shape
        _, N_k, _ = k_coords.shape
        k = min(self.k, N_k)

        # Precompute k_sq under no_grad to save autograd graph memory
        with torch.no_grad():
            k_sq = torch.sum(k_coords ** 2, dim=-1, keepdim=True)       # (B, N_k, 1)

        out_list = []
        for i in range(0, N_q, chunk_size):
            q_c_chunk = q_coords[:, i : i + chunk_size, :]              # (B, nc, 3)
            q_f_chunk = q_feats[:, i : i + chunk_size, :]               # (B, nc, C)

            # 1. Compute chunk pairwise distance & find kNN under no_grad
            # Nearest neighbor indexing is non-differentiable anyway, so no_grad prevents
            # storing the large distance matrix in PyTorch's backward graph.
            with torch.no_grad():
                q_sq_chunk = torch.sum(q_c_chunk ** 2, dim=-1, keepdim=True) # (B, nc, 1)
                qk_chunk = torch.bmm(q_c_chunk, k_coords.transpose(1, 2))   # (B, nc, N_k)
                dist_chunk = q_sq_chunk + k_sq.transpose(1, 2) - 2.0 * qk_chunk
                _, knn_idx = torch.topk(-dist_chunk, k=k, dim=-1)           # (B, nc, k)

            # 2. Gather neighbor coordinates and features
            neighbor_coords = knn_gather(k_coords, knn_idx)                 # (B, nc, k, 3)
            neighbor_feats = knn_gather(k_feats, knn_idx)                   # (B, nc, k, C)

            # 3. Relative position encoding
            delta_p = neighbor_coords - q_c_chunk.unsqueeze(2)              # (B, nc, k, 3)
            pos_enc = self.pos_mlp(delta_p)                                 # (B, nc, k, C)

            # 4. Attention
            Q = self.q_proj(q_f_chunk).unsqueeze(2)                         # (B, nc, 1, C)
            feat_pos = neighbor_feats + pos_enc
            K = self.k_proj(feat_pos)                                       # (B, nc, k, C)
            V = self.v_proj(feat_pos)                                       # (B, nc, k, C)

            attn = torch.sum(Q * K, dim=-1, keepdim=True) / (self.channels ** 0.5)  # (B, nc, k, 1)
            attn = F.softmax(attn, dim=2)                                   # (B, nc, k, 1)

            aggregated = torch.sum(attn * V, dim=2)                         # (B, nc, C)
            out_chunk = self.out_proj(aggregated)                           # (B, nc, C)
            out_list.append(out_chunk)

        out = torch.cat(out_list, dim=1) if len(out_list) > 1 else out_list[0]
        return self.norm(q_feats + out)


class PointUpsampleBlock(nn.Module):
    """Upsample block: Local kNN Attention -> Upsample features -> 3-layer offset prediction.
    - factor: upsampling factor (e.g. 2 or 4)
    """
    def __init__(self, in_channels: int = 128, factor: int = 2, k: int = 16):
        super().__init__()
        self.factor = factor
        self.attention = LocalKNNSelfAttention(channels=in_channels, k=k)

        # Feature upsampling via linear expansion
        self.feat_upsample = nn.Sequential(
            nn.Linear(in_channels, in_channels * factor),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * factor, in_channels * factor)
        )

        # 3 Convolutional/MLP layers to predict 3D offsets for upsampled points
        self.offset_net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(
        self,
        p_coords: torch.Tensor,
        p_feats: torch.Tensor,
        ctx_coords: torch.Tensor,
        ctx_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p_coords: (B, N, 3)
            p_feats:  (B, N, C)
            ctx_coords: (B, M, 3) combined object + hand points
            ctx_feats:  (B, M, C) combined object + hand features
        Returns:
            up_coords: (B, N * factor, 3)
            up_feats:  (B, N * factor, C)
        """
        B, N, C = p_feats.shape

        # 1. Local context self-attention
        attended_feats = self.attention(p_coords, p_feats, ctx_coords, ctx_feats)  # (B, N, C)

        # 2. Upsample features
        # (B, N, C * factor) -> (B, N * factor, C)
        up_feats_raw = self.feat_upsample(attended_feats)                          # (B, N, C * factor)
        up_feats = up_feats_raw.view(B, N * self.factor, C)                        # (B, N * factor, C)

        # 3. Base point replication
        base_coords = p_coords.repeat_interleave(self.factor, dim=1)               # (B, N * factor, 3)

        # 4. Predict 3D offsets
        offsets = self.offset_net(up_feats)                                        # (B, N * factor, 3)
        up_coords = base_coords + offsets                                          # (B, N * factor, 3)

        return up_coords, up_feats


class DenseDecoder(nn.Module):
    """Dense Point Cloud Decoder from paper Section 3.4.
    - Removes DINOv2 CLS token
    - Reshapes 256 patch tokens to 16x16 spatial grid
    - 3x3 convolutions -> 16x16x128 feature map f_v^r
    - Transforms sparse points to camera coordinates: p_s + t_p + t_o
    - Projects via camera intrinsics K and bilinearly samples pixel-aligned features
    - Retrieves pixel-aligned features for hand vertices too
    - 2 progressive upsampling blocks:
        Block 1: x2 upsampling (2048 -> 4096 points)
        Block 2: x4 upsampling (4096 -> 16,384 points)
    - Output: dense 3D point cloud of 16,384 points
    """
    def __init__(
        self,
        image_feat_dim: int = 1024,
        feat_map_dim: int = 128,
        k_knn: int = 16,
        first_upsample_factor: int = 2,
        second_upsample_factor: int = 4,
        num_dense_points: int = 16384,
        img_size: int = 224
    ):
        super().__init__()
        self.feat_map_dim = feat_map_dim
        self.img_size = img_size
        self.num_dense_points = num_dense_points

        # 3x3 Convolutions to refine spatial patch tokens: (B, 1024, 16, 16) -> (B, 128, 16, 16)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(image_feat_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, feat_map_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_map_dim),
            nn.ReLU(inplace=True)
        )

        # Coordinate + visual feature projection: (3 + 128 = 131) -> 128
        self.coord_feat_proj = nn.Sequential(
            nn.Linear(3 + feat_map_dim, feat_map_dim),
            nn.LayerNorm(feat_map_dim),
            nn.ReLU(inplace=True)
        )

        # Progressive upsampling blocks
        self.block1 = PointUpsampleBlock(in_channels=feat_map_dim, factor=first_upsample_factor, k=k_knn)
        self.block2 = PointUpsampleBlock(in_channels=feat_map_dim, factor=second_upsample_factor, k=k_knn)

    def _project_and_sample_features(
        self,
        p_cam: torch.Tensor,
        feat_map: torch.Tensor,
        cam_intr: torch.Tensor
    ) -> torch.Tensor:
        """Projects 3D points into image plane and samples visual features.

        Steps:
          1. Perspective projection: u = fx*(X/Z)+cx, v = fy*(Y/Z)+cy
          2. Normalize (u,v) to [-1,1] for grid_sample
          3. Bilinear sample from feat_map (B, C, 16, 16)

        Args:
            p_cam:    (B, N, 3) 3D points in camera coordinates
            feat_map: (B, 128, 16, 16) spatial image feature map
            cam_intr: (B, 3, 3) camera intrinsics [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        Returns:
            aligned_feats: (B, N, 128) per-point pixel-aligned features
        """
        # Clamp depth to avoid division by zero
        Z = torch.clamp(p_cam[:, :, 2], min=1e-4)                   # (B, N)
        # Project to 2D pixel coordinates
        u = cam_intr[:, 0:1, 0] * (p_cam[:, :, 0] / Z) + cam_intr[:, 0:1, 2]  # (B, N)
        v = cam_intr[:, 1:2, 1] * (p_cam[:, :, 1] / Z) + cam_intr[:, 1:2, 2]  # (B, N)

        # Normalize to [-1, 1] for grid_sample (expects x=col, y=row convention)
        scale = float(self.img_size - 1)
        grid = torch.stack([2.0 * u / scale - 1.0, 2.0 * v / scale - 1.0], dim=-1)  # (B, N, 2)
        grid = grid.unsqueeze(2)  # (B, N, 1, 2) — grid_sample needs 4D grid

        # Bilinear sample: output is (B, C, N, 1) -> squeeze and transpose to (B, N, C)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)                   # (B, N, 128)

    def forward(
        self,
        p_s: torch.Tensor,
        to: torch.Tensor,
        palm_coord: torch.Tensor,
        fv: torch.Tensor,
        hand_verts: torch.Tensor,
        cam_intr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p_s: (B, 2048, 3) sparse object point cloud
            to: (B, 3) predicted hand-relative object translation
            palm_coord: (B, 3) hand palm 3D position
            fv: (B, 257, 1024) image feature tokens
            hand_verts: (B, 778, 3) hand vertices (camera space)
            cam_intr: (B, 3, 3) camera intrinsics
        Returns:
            p_dense: (B, 16384, 3) dense 3D point cloud
        """
        B = p_s.shape[0]

        # 1. Remove [CLS] token (index 0) and reshape 256 patch tokens to spatial grid 16x16
        # fv: (B, 257, 1024) -> patch_tokens: (B, 256, 1024)
        patch_tokens = fv[:, 1:, :]                                                # (B, 256, 1024)
        # Reshape to (B, 1024, 16, 16)
        patch_grid = patch_tokens.transpose(1, 2).view(B, -1, 16, 16)             # (B, 1024, 16, 16)

        # 2. 3x3 Convolutions to obtain refined feature map f_v^r: (B, 128, 16, 16)
        f_v_r = self.conv_refine(patch_grid)                                       # (B, 128, 16, 16)

        # 3. Transform sparse points to camera coordinates:
        # P_cam = p_s + t_p + t_o
        shift = (palm_coord + to).unsqueeze(1)                                     # (B, 1, 3)
        p_cam_obj = p_s + shift                                                    # (B, 2048, 3)

        # 4. Bilinear sampling of pixel-aligned features
        f_obj = self._project_and_sample_features(p_cam_obj, f_v_r, cam_intr)      # (B, 2048, 128)
        f_hand = self._project_and_sample_features(hand_verts, f_v_r, cam_intr)    # (B, 778, 128)

        # 5. Concatenate point coordinates with aligned visual features and project
        # Object points: [p_s, f_obj] -> (B, 2048, 131) -> (B, 2048, 128)
        obj_in = torch.cat([p_s, f_obj], dim=-1)
        obj_feat = self.coord_feat_proj(obj_in)                                    # (B, 2048, 128)

        # Hand context relative to palm: [v_h - palm, f_hand] -> (B, 778, 131) -> (B, 778, 128)
        hand_rel_coords = hand_verts - palm_coord.unsqueeze(1)
        hand_in = torch.cat([hand_rel_coords, f_hand], dim=-1)
        hand_feat = self.coord_feat_proj(hand_in)                                  # (B, 778, 128)

        # Combined context (object + hand)
        ctx_coords = torch.cat([p_s, hand_rel_coords], dim=1)                     # (B, 2048 + 778, 3)
        ctx_feats = torch.cat([obj_feat, hand_feat], dim=1)                        # (B, 2048 + 778, 128)

        # 6. Progressive upsampling
        # Block 1: x2 upsampling (2048 -> 4096 points)
        if self.training and p_s.requires_grad:
            try:
                p_coords_1, p_feats_1 = torch.utils.checkpoint.checkpoint(
                    self.block1, p_s, obj_feat, ctx_coords, ctx_feats, use_reentrant=False
                )
            except TypeError:
                p_coords_1, p_feats_1 = torch.utils.checkpoint.checkpoint(
                    self.block1, p_s, obj_feat, ctx_coords, ctx_feats
                )
        else:
            p_coords_1, p_feats_1 = self.block1(p_s, obj_feat, ctx_coords, ctx_feats) # (B, 4096, 3), (B, 4096, 128)

        # Update context with block 1 points
        ctx_coords_1 = torch.cat([p_coords_1, hand_rel_coords], dim=1)            # (B, 4096 + 778, 3)
        ctx_feats_1 = torch.cat([p_feats_1, hand_feat], dim=1)                     # (B, 4096 + 778, 128)

        # Block 2: x4 upsampling (4096 -> 16,384 points)
        if self.training and p_coords_1.requires_grad:
            try:
                p_dense, _ = torch.utils.checkpoint.checkpoint(
                    self.block2, p_coords_1, p_feats_1, ctx_coords_1, ctx_feats_1, use_reentrant=False
                )
            except TypeError:
                p_dense, _ = torch.utils.checkpoint.checkpoint(
                    self.block2, p_coords_1, p_feats_1, ctx_coords_1, ctx_feats_1
                )
        else:
            p_dense, _ = self.block2(p_coords_1, p_feats_1, ctx_coords_1, ctx_feats_1)# (B, 16384, 3)

        assert p_dense.shape[1] == self.num_dense_points, \
            f"Expected {self.num_dense_points} dense points, got {p_dense.shape[1]}"
        return p_dense

