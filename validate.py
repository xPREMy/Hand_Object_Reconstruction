import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.hort import HORT
from losses.chamfer import chamfer_distance_chunked
from train import build_model, build_dataset, load_config


def compute_f_score(
    p_pred: torch.Tensor,
    p_gt: torch.Tensor,
    threshold: float
) -> float:
    """Computes F-score at distance threshold (in meters).
    Args:
        p_pred: (N, 3)
        p_gt:   (M, 3)
        threshold: distance threshold (e.g. 0.005 for 5mm, 0.010 for 10mm)
    Returns:
        f_score: scalar in [0, 1]
    """
    # Precision: p_pred -> p_gt
    # (N, 1, 3) - (1, M, 3)
    diff_p = p_pred.unsqueeze(1) - p_gt.unsqueeze(0)             # (N, M, 3)
    dist_p = torch.norm(diff_p, dim=-1)                          # (N, M)
    min_dist_p, _ = torch.min(dist_p, dim=-1)                    # (N,)
    precision = torch.mean((min_dist_p < threshold).float()).item()

    # Recall: p_gt -> p_pred
    min_dist_q, _ = torch.min(dist_p, dim=0)                     # (M,)
    recall = torch.mean((min_dist_q < threshold).float()).item()

    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def validate(
    model: HORT,
    loader: DataLoader,
    device: torch.device
) -> dict:
    model.eval()
    chamfer_denses = []
    chamfer_sparses = []
    pose_errors = []
    f_scores_5 = []
    f_scores_10 = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            hand_verts = batch["hand_verts"].to(device)
            hand_joints = batch["hand_joints"].to(device)
            palm_coord = batch["palm_coord"].to(device)
            cam_intr = batch["cam_intr"].to(device)
            gt_trans = batch["gt_trans"].to(device)
            gt_sparse = batch["gt_points_sparse"].to(device)
            gt_dense = batch["gt_points_dense"].to(device)

            preds = model(
                image=images,
                hand_verts=hand_verts,
                hand_joints=hand_joints,
                palm_coord=palm_coord,
                cam_intr=cam_intr
            )

            pred_trans = preds["pred_trans"]
            pred_sparse = preds["sparse_points"]
            pred_dense = preds["dense_points"]

            # CD losses
            cd_sparse = chamfer_distance_chunked(pred_sparse, gt_sparse).item()
            cd_dense = chamfer_distance_chunked(pred_dense, gt_dense).item()
            pose_l1 = torch.mean(torch.abs(pred_trans - gt_trans)).item()

            chamfer_sparses.append(cd_sparse)
            chamfer_denses.append(cd_dense)
            pose_errors.append(pose_l1)

            # Compute F-scores per sample
            B = images.shape[0]
            for b in range(B):
                fs5 = compute_f_score(pred_dense[b], gt_dense[b], threshold=0.005)   # 5mm
                fs10 = compute_f_score(pred_dense[b], gt_dense[b], threshold=0.010) # 10mm
                f_scores_5.append(fs5)
                f_scores_10.append(fs10)

    # In paper, CD is in cm^2: 1 m^2 = 10000 cm^2
    mean_cd_cm2 = np.mean(chamfer_denses) * 10000.0
    mean_fs5 = np.mean(f_scores_5)
    mean_fs10 = np.mean(f_scores_10)
    mean_pose_cm = np.mean(pose_errors) * 100.0

    return {
        "cd_dense_cm2": mean_cd_cm2,
        "fs@5": mean_fs5,
        "fs@10": mean_fs10,
        "pose_error_cm": mean_pose_cm,
        "mean_sparse_cd": np.mean(chamfer_sparses)
    }


def main():
    parser = argparse.ArgumentParser(description="Validate HORT")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.get("dataset", {}).get("name", "obman")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Validate] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = build_dataset(cfg, dataset_name, split=args.split, max_samples=args.max_samples)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[Validate] Evaluating on {len(val_dataset)} samples ({args.split} split)...")
    results = validate(model, val_loader, device)

    print("\n" + "=" * 50)
    print(f"HORT Evaluation Results on {dataset_name.upper()} ({args.split}):")
    print("=" * 50)
    print(f"FS@5mm  (↑)  : {results['fs@5']:.4f}")
    print(f"FS@10mm (↑)  : {results['fs@10']:.4f}")
    print(f"CD (cm²) (↓) : {results['cd_dense_cm2']:.4f}")
    print(f"Pose L1 (cm) : {results['pose_error_cm']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

