import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_dist_one_direction(src: torch.Tensor, tgt: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """For each point in src, find squared distance to nearest point in tgt.
    Args:
        src: (B, N, 3)
        tgt: (B, M, 3)
    Returns:
        min_dists: (B, N) minimum squared distances
    """
    tgt_sq = torch.sum(tgt ** 2, dim=-1, keepdim=True)  # (B, M, 1)
    result = []
    for i in range(0, src.shape[1], chunk_size):
        chunk = src[:, i : i + chunk_size, :]                        # (B, nc, 3)
        chunk_sq = torch.sum(chunk ** 2, dim=-1, keepdim=True)       # (B, nc, 1)
        # squared distance between each chunk point and every tgt point
        dist = chunk_sq + tgt_sq.transpose(1, 2) - 2.0 * torch.bmm(chunk, tgt.transpose(1, 2))
        dist = torch.clamp(dist, min=0.0)                            # (B, nc, M)
        min_d, _ = torch.min(dist, dim=-1)                           # (B, nc)
        result.append(min_d)
    return torch.cat(result, dim=1)                                  # (B, N)


def chamfer_distance_chunked(p1: torch.Tensor, p2: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
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
    def __init__(self, chunk_size: int = 2048):
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
        chunk_size: int = 2048,
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
