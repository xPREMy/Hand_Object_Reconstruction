import torch
import torch.nn as nn


# MANO fingertip vertex indices (standard MANO topology)
# Thumb, Index, Middle, Ring, Pinky
MANO_FINGERTIP_VERT_INDICES = [745, 333, 444, 555, 672]  # approximate tip vertices


class HandEncoder(nn.Module):
    """Hand Geometry Encoder from paper Section 3.2.
    - Input:
        hand_verts: (B, 778, 3) 3D hand vertices
        hand_joints: (B, 16, 3) or (B, 21, 3) 3D hand joints
        palm_coord: (B, 3) 3D palm coordinate
    - Creates 22 local coordinate systems:
        16 joints + 5 fingertips + 1 palm
    - Transforms every vertex into each coordinate system: (B, 778, 66)
    - Appends normalized vertex index: (B, 778, 67)
    - 5-layer PointNet MLP -> Global max pooling -> (B, 1024) hand feature
    """
    def __init__(
        self,
        num_verts: int = 778,
        num_coords: int = 22,
        in_dim: int = 67,
        out_dim: int = 1024
    ):
        super().__init__()
        self.num_verts = num_verts
        self.num_coords = num_coords
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 5 MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

        # Pre-computed normalized vertex index buffer
        vert_idx = torch.arange(num_verts, dtype=torch.float32) / float(max(1, num_verts - 1))
        self.register_buffer("vert_idx", vert_idx.unsqueeze(-1))  # (778, 1)

    def _assemble_22_coordinate_centers(
        self,
        hand_verts: torch.Tensor,
        hand_joints: torch.Tensor,
        palm_coord: torch.Tensor
    ) -> torch.Tensor:
        """Assembles the 22 coordinate system origins:
        - 16 joints
        - 5 fingertips
        - 1 palm
        Returns:
            centers: (B, 22, 3)
        """
        B = hand_verts.shape[0]

        if hand_joints.shape[1] == 21:
            # First 16 are base/intermediate joints, last 5 are fingertips
            joints_16 = hand_joints[:, :16, :]  # (B, 16, 3)
            tips_5 = hand_joints[:, 16:21, :]   # (B, 5, 3)
        elif hand_joints.shape[1] == 16:
            joints_16 = hand_joints             # (B, 16, 3)
            # Sample fingertip vertices from hand_verts
            tip_indices = torch.tensor(MANO_FINGERTIP_VERT_INDICES, device=hand_verts.device)
            tips_5 = hand_verts[:, tip_indices, :]  # (B, 5, 3)
        else:
            raise ValueError(f"Unexpected hand_joints shape: {hand_joints.shape}, expected 16 or 21 joints")

        palm = palm_coord.unsqueeze(1)  # (B, 1, 3)
        centers = torch.cat([joints_16, tips_5, palm], dim=1)  # (B, 22, 3)
        assert centers.shape[1] == 22, f"Expected 22 coordinate centers, got {centers.shape[1]}"
        return centers

    def forward(
        self,
        hand_verts: torch.Tensor,
        hand_joints: torch.Tensor,
        palm_coord: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hand_verts: (B, 778, 3)
            hand_joints: (B, 16, 3) or (B, 21, 3)
            palm_coord: (B, 3)
        Returns:
            fh: (B, 1024) hand feature vector
        """
        B, V, _ = hand_verts.shape
        assert V == self.num_verts, f"Expected {self.num_verts} vertices, got {V}"

        # 1. 22 coordinate centers (B, 22, 3)
        centers = self._assemble_22_coordinate_centers(hand_verts, hand_joints, palm_coord)

        # 2. Transform vertices into each of the 22 coordinate systems
        # hand_verts: (B, 778, 1, 3) - centers: (B, 1, 22, 3) -> (B, 778, 22, 3)
        diff = hand_verts.unsqueeze(2) - centers.unsqueeze(1)  # (B, 778, 22, 3)
        diff = diff.reshape(B, V, 22 * 3)                      # (B, 778, 66)

        # 3. Concatenate absolute vertex index: (B, 778, 67)
        idx = self.vert_idx.unsqueeze(0).expand(B, -1, -1)     # (B, 778, 1)
        eh = torch.cat([diff, idx], dim=-1)                    # (B, 778, 67)

        # 4. PointNet 5-layer MLP
        feat = self.mlp(eh)                                    # (B, 778, 1024)

        # 5. Global max pooling over vertices
        fh, _ = torch.max(feat, dim=1)                         # (B, 1024)
        return fh

