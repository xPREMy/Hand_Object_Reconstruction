import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_dist_one_direction(src: torch.Tensor, tgt: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """For each point in src, find squared distance to nearest point in tgt.
    Memory-efficient: nearest neighbor indices are computed under no_grad so
    the large pairwise distance matrices (B, chunk_size, M) are never retained
    in the autograd graph. Gradients flow directly through gathered matched points.
    
    Args:
        src: (B, N, 3)
        tgt: (B, M, 3)
        chunk_size: number of points to process per chunk to minimize peak VRAM
    Returns:
        min_dists: (B, N) minimum squared distances
    """
    B, N, _ = src.shape
    M = tgt.shape[1]

    # Find nearest neighbor indices under no_grad (frees chunk tensors immediately)
    idx_list = []
    with torch.no_grad():
        tgt_sq = torch.sum(tgt ** 2, dim=-1, keepdim=True)  # (B, M, 1)
        for i in range(0, N, chunk_size):
            chunk = src[:, i : i + chunk_size, :]                        # (B, nc, 3)
            chunk_sq = torch.sum(chunk ** 2, dim=-1, keepdim=True)       # (B, nc, 1)
            dist = chunk_sq + tgt_sq.transpose(1, 2) - 2.0 * torch.bmm(chunk, tgt.transpose(1, 2))
            min_idx = torch.argmin(dist, dim=-1)                         # (B, nc)
            idx_list.append(min_idx)

    nearest_idx = torch.cat(idx_list, dim=1)                             # (B, N)
    # Gather matched points: (B, N, 3)
    matched_tgt = torch.gather(tgt, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, 3))
    # Exact squared Euclidean distance with O(N) backward memory instead of O(N*M)
    min_dists = torch.sum((src - matched_tgt) ** 2, dim=-1)              # (B, N)
    return min_dists


def chamfer_distance_chunked(p1: torch.Tensor, p2: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Memory-efficient bidirectional Chamfer Distance (mean of squared Euclidean distances).

    Why chunked? Computing (B, N, M) distance matrix at once for N=16384 causes OOM.
    We process 'chunk_size' points at a time.

    Args:
        p1: (B, N, 3) predicted point cloud
        p2: (B, M, 3) ground-truth point cloud
    Returns:
        scalar: mean Chamfer distance across batch
    """
    d1 = _min_dist_one_direction(p1, p2, chunk_size)  # (B, N): each pred -> nearest GT
    d2 = _min_dist_one_direction(p2, p1, chunk_size)  # (B, M): each GT -> nearest pred
    cd = d1.mean(dim=1) + d2.mean(dim=1)              # (B,)
    return cd.mean()


class ChamferLoss(nn.Module):
    """Wraps chamfer_distance_chunked as an nn.Module."""
    def __init__(self, chunk_size: int = 512):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return chamfer_distance_chunked(pred, gt, self.chunk_size)


class HORTLoss(nn.Module):
    """Composite HORT training loss (paper Section 3.5):

        L = lambda_pose * L_pose + lambda_sparse * L_cd_sparse + L_cd_dense

    where:
        L_pose      = L1 loss on predicted hand-relative 3D object translation
        L_cd_sparse = Chamfer Distance between 2048 sparse predicted and GT points
        L_cd_dense  = Chamfer Distance between 16384 dense predicted and GT points
        lambda_pose = 2,  lambda_sparse = 2  (paper defaults)
    """
    def __init__(
        self,
        lambda_pose: float = 2.0,
        lambda_sparse_cd: float = 2.0,
        lambda_dense_cd: float = 1.0,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_sparse_cd = lambda_sparse_cd
        self.lambda_dense_cd = lambda_dense_cd
        self.chamfer = ChamferLoss(chunk_size)

    def forward(
        self,
        pred_trans: torch.Tensor,        # (B, 3)
        gt_trans: torch.Tensor,          # (B, 3)
        pred_sparse_points: torch.Tensor,# (B, 2048, 3)
        gt_sparse_points: torch.Tensor,  # (B, 2048, 3)
        pred_dense_points: torch.Tensor, # (B, 16384, 3)
        gt_dense_points: torch.Tensor,   # (B, 16384, 3)
    ) -> dict:
        loss_pose      = F.l1_loss(pred_trans, gt_trans)
        loss_sparse_cd = self.chamfer(pred_sparse_points, gt_sparse_points)
        loss_dense_cd  = self.chamfer(pred_dense_points,  gt_dense_points)

        total = (
            self.lambda_pose      * loss_pose
            + self.lambda_sparse_cd * loss_sparse_cd
            + self.lambda_dense_cd  * loss_dense_cd
        )
        return {
            "loss":           total,
            "loss_pose":      loss_pose,
            "loss_sparse_cd": loss_sparse_cd,
            "loss_dense_cd":  loss_dense_cd,
        }
