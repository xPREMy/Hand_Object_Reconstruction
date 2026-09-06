import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .hand_encoder import HandEncoder
from .sparse_decoder import SparseDecoder
from .dense_decoder import DenseDecoder


class HORT(nn.Module):
    """HORT: Monocular Hand-held Objects Reconstruction with Transformers (ICCV 2025).
    Architecture:
        1. Image Encoder: DINOv2-Large (257x1024 tokens)
        2. Hand Encoder: 22 coordinate systems PointNet (1024-D feature)
        3. Sparse Decoder: 10-layer Transformer jointly decoding hand-relative translation t_o
           and 2048 sparse points p_s
        4. Dense Decoder: 2-block kNN self-attention + progressive upsampling (x2, x4)
           to 16,384 dense points p_d using pixel-aligned features
    """
    def __init__(
        self,
        # Image encoder config
        backbone_name: str = "vit_large_patch14_dinov2",
        pretrained: bool = True,
        img_size: int = 224,
        embed_dim: int = 1024,
        num_frozen_layers: int = 12,
        # Hand encoder config
        num_hand_verts: int = 778,
        num_coords: int = 22,
        hand_in_dim: int = 67,
        hand_feat_dim: int = 1024,
        # Sparse decoder config
        sparse_hidden_dim: int = 512,
        sparse_num_layers: int = 10,
        sparse_num_heads: int = 8,
        sparse_ffn_dim: int = 2048,
        num_sparse_points: int = 2048,
        # Dense decoder config
        dense_feat_map_dim: int = 128,
        dense_k_knn: int = 16,
        first_upsample_factor: int = 2,
        second_upsample_factor: int = 4,
        num_dense_points: int = 16384
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            embed_dim=embed_dim,
            num_frozen_layers=num_frozen_layers
        )

        self.hand_encoder = HandEncoder(
            num_verts=num_hand_verts,
            num_coords=num_coords,
            in_dim=hand_in_dim,
            out_dim=hand_feat_dim
        )

        self.sparse_decoder = SparseDecoder(
            hidden_dim=sparse_hidden_dim,
            num_layers=sparse_num_layers,
            num_heads=sparse_num_heads,
            ffn_dim=sparse_ffn_dim,
            num_sparse_points=num_sparse_points,
            image_feat_dim=embed_dim,
            hand_feat_dim=hand_feat_dim
        )

        self.dense_decoder = DenseDecoder(
            image_feat_dim=embed_dim,
            feat_map_dim=dense_feat_map_dim,
            k_knn=dense_k_knn,
            first_upsample_factor=first_upsample_factor,
            second_upsample_factor=second_upsample_factor,
            num_dense_points=num_dense_points,
            img_size=img_size
        )

    def forward(
        self,
        image: torch.Tensor,
        hand_verts: torch.Tensor,
        hand_joints: torch.Tensor,
        palm_coord: torch.Tensor,
        cam_intr: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, 224, 224) RGB image normalized
            hand_verts: (B, 778, 3) MANO hand vertices in camera coordinates
            hand_joints: (B, 16, 3) or (B, 21, 3) MANO joints in camera coordinates
            palm_coord: (B, 3) hand palm coordinate in camera coordinates
            cam_intr: (B, 3, 3) camera intrinsics matrix
        Returns:
            dict containing:
                pred_trans: (B, 3) hand-relative 3D object translation t_o
                sparse_points: (B, 2048, 3) sparse object point cloud p_s
                dense_points: (B, 16384, 3) dense object point cloud p_d
                image_features: (B, 257, 1024)
                hand_feature: (B, 1024)
        """
        # 1. Image features: f_v in R^(B x 257 x 1024)
        fv = self.image_encoder(image)

        # 2. Hand features: f_h in R^(B x 1024)
        fh = self.hand_encoder(hand_verts, hand_joints, palm_coord)

        # 3. Sparse prediction: t_o in R^(B x 3), p_s in R^(B x 2048 x 3)
        to, p_s = self.sparse_decoder(fv, fh)

        # 4. Dense reconstruction: p_d in R^(B x 16384 x 3)
        p_d = self.dense_decoder(
            p_s=p_s,
            to=to,
            palm_coord=palm_coord,
            fv=fv,
            hand_verts=hand_verts,
            cam_intr=cam_intr
        )

        return {
            "pred_trans": to,
            "sparse_points": p_s,
            "dense_points": p_d,
            "image_features": fv,
            "hand_feature": fh,
        }

