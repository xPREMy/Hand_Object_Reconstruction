import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.hort import HORT
import json
import trimesh
from losses.chamfer import chamfer_distance_chunked
from train import build_model, build_dataset, load_config


def export_point_cloud(points: np.ndarray, file_path: str):
    """Save point cloud as .ply."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(file_path, pcd)
    except Exception:
        pcd = trimesh.PointCloud(vertices=points)
        pcd.export(file_path)


def export_mesh(points: np.ndarray, file_path: str):
    """Generate surface mesh via Open3D Poisson Reconstruction or Convex Hull fallback."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o3d.io.write_triangle_mesh(file_path, mesh)
    except Exception:
        hull = trimesh.convex.convex_hull(points)
        hull.export(file_path)


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
    diff_p = p_pred.unsqueeze(1) - p_gt.unsqueeze(0)             # (N, M, 3)
    dist_p = torch.norm(diff_p, dim=-1)                          # (N, M)
    min_dist_p, _ = torch.min(dist_p, dim=-1)                    # (N,)
    precision = torch.mean((min_dist_p < threshold).float()).item()

    min_dist_q, _ = torch.min(dist_p, dim=0)                     # (M,)
    recall = torch.mean((min_dist_q < threshold).float()).item()

    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def validate(
    model: HORT,
    loader: DataLoader,
    device: torch.device,
    save_dir: str | None = None,
    max_save_models: int = 20,
    save_mesh: bool = True
) -> dict:
    model.eval()
    chamfer_denses = []
    chamfer_sparses = []
    pose_errors = []
    f_scores_5 = []
    f_scores_10 = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    saved_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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

                # Export 3D models if requested
                if save_dir and saved_count < max_save_models:
                    sample_id = f"sample_{saved_count:03d}"
                    palm_np = palm_coord[b].cpu().numpy()
                    to_np = pred_trans[b].cpu().numpy()
                    
                    # Convert to camera frame: P_cam = P_local + palm + to
                    pred_sparse_cam = pred_sparse[b].cpu().numpy() + palm_np + to_np
                    pred_dense_cam = pred_dense[b].cpu().numpy() + palm_np + to_np
                    hand_cam = hand_verts[b].cpu().numpy()
                    
                    # Save PLY point clouds
                    sparse_path = os.path.join(save_dir, f"{sample_id}_pred_sparse.ply")
                    dense_path = os.path.join(save_dir, f"{sample_id}_pred_dense.ply")
                    hand_path = os.path.join(save_dir, f"{sample_id}_hand.ply")
                    
                    export_point_cloud(pred_sparse_cam, sparse_path)
                    export_point_cloud(pred_dense_cam, dense_path)
                    export_point_cloud(hand_cam, hand_path)
                    
                    # Ground truth dense cloud
                    gt_to_np = gt_trans[b].cpu().numpy()
                    gt_dense_cam = gt_dense[b].cpu().numpy() + palm_np + gt_to_np
                    gt_dense_path = os.path.join(save_dir, f"{sample_id}_gt_dense.ply")
                    export_point_cloud(gt_dense_cam, gt_dense_path)

                    # Surface mesh
                    if save_mesh:
                        mesh_path = os.path.join(save_dir, f"{sample_id}_dense_mesh.ply")
                        try:
                            export_mesh(pred_dense_cam, mesh_path)
                        except Exception:
                            pass
                    
                    saved_count += 1

    mean_cd_cm2 = np.mean(chamfer_denses) * 10000.0
    mean_fs5 = np.mean(f_scores_5)
    mean_fs10 = np.mean(f_scores_10)
    mean_pose_cm = np.mean(pose_errors) * 100.0

    results = {
        "cd_dense_cm2": float(mean_cd_cm2),
        "fs@5": float(mean_fs5),
        "fs@10": float(mean_fs10),
        "pose_error_cm": float(mean_pose_cm),
        "mean_sparse_cd": float(np.mean(chamfer_sparses)),
        "saved_3d_models_count": saved_count
    }

    if save_dir:
        summary_file = os.path.join(save_dir, "evaluation_summary.json")
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Validate] Saved 3D models and evaluation summary to: {save_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate HORT")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save 3D models and metrics JSON")
    parser.add_argument("--max_save_models", type=int, default=20, help="Maximum number of 3D models to export")
    parser.add_argument("--no_mesh", action="store_true", help="Disable surface mesh reconstruction")
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
    results = validate(
        model=model,
        loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        max_save_models=args.max_save_models,
        save_mesh=not args.no_mesh
    )

    print("\n" + "=" * 50)
    print(f"HORT Evaluation Results on {dataset_name.upper()} ({args.split}):")
    print("=" * 50)
    print(f"FS@5mm  (↑)  : {results['fs@5']:.4f}")
    print(f"FS@10mm (↑)  : {results['fs@10']:.4f}")
    print(f"CD (cm²) (↓) : {results['cd_dense_cm2']:.4f}")
    print(f"Pose L1 (cm) : {results['pose_error_cm']:.4f}")
    if args.save_dir:
        print(f"3D Models    : Exported to {args.save_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()

