import os
import argparse
import json
import numpy as np
import torch
import cv2
import trimesh
import open3d as o3d

from models.hort import HORT
from train import build_model, load_config
from datasets.base import BaseHandObjectDataset


def export_point_cloud_ply(points: np.ndarray, file_path: str, colors: np.ndarray | None = None):
    """Exports (N, 3) points to .ply file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(file_path, pcd)


def export_mesh_ply(vertices: np.ndarray, faces: np.ndarray, file_path: str):
    """Exports vertices and faces to .ply / .obj mesh file."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(file_path)


def generate_mesh_from_point_cloud(points: np.ndarray, file_path: str):
    """Generates surface mesh from dense point cloud using Open3D Poisson Reconstruction."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    o3d.io.write_triangle_mesh(file_path, mesh)


def get_default_hand_mesh(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Provides a default neutral hand mesh (778 verts, 21 joints, palm) in camera space.
    [APPROXIMATION when external hand pose estimator WiLoR output is not supplied].
    """
    # Canonical hand mesh approximation positioned at typical distance (0.6m in front of camera)
    palm = torch.tensor([0.0, 0.05, 0.60], dtype=torch.float32, device=device)

    # Generate canonical open hand shape (778 vertices, MANO topology)
    # Load default MANO if available via smplx or generate canonical hand shape
    try:
        import smplx
        # Attempt minimal smplx MANO instantiate if weights available
        hand_model = smplx.create(
            model_path="",
            model_type="mano",
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True
        )
        output = hand_model()
        verts = output.vertices[0].to(device) + palm
        joints = output.joints[0, :21].to(device) + palm
    except Exception:
        # Fallback procedural template hand mesh
        theta = torch.linspace(0, 2 * np.pi, 778, device=device)
        xs = 0.04 * torch.cos(theta) * torch.sin(theta * 2)
        ys = 0.08 * torch.sin(theta)
        zs = 0.02 * torch.cos(theta * 3)
        verts = torch.stack([xs, ys, zs], dim=-1) + palm
        joints = verts[:21] + torch.tensor([0.0, 0.0, -0.01], device=device)

    return verts, joints, palm


def main():
    parser = argparse.ArgumentParser(description="HORT Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="output_hort", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--hand_mesh", type=str, default=None, help="Optional path to estimated hand mesh (.ply/.obj/.pkl)")
    parser.add_argument("--generate_mesh", action="store_true", default=True, help="Generate surface mesh from dense cloud")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Using device: {device}")

    # 1. Load image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 2. Camera Intrinsics
    # When camera intrinsics are not provided at test time, use standard perspective default
    fx = fy = float(max(H, W))
    cx = float(W / 2.0)
    cy = float(H / 2.0)
    cam_intr_raw = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # 3. Hand Mesh Input
    if args.hand_mesh and os.path.exists(args.hand_mesh):
        mesh = trimesh.load(args.hand_mesh, force="mesh")
        verts = torch.from_numpy(mesh.vertices[:778]).float().to(device)
        palm = torch.mean(verts, dim=0)
        joints = verts[:21]
    else:
        # Default estimated hand mesh [APPROXIMATION]
        verts, joints, palm = get_default_hand_mesh(device)

    # 4. Hand-Object Bounding Box & Preprocessing
    # Project hand to compute crop bbox
    verts_np = verts.detach().cpu().numpy()
    zs = np.maximum(verts_np[:, 2], 1e-4)
    xs = (verts_np[:, 0] / zs) * fx + cx
    ys = (verts_np[:, 1] / zs) * fy + cy
    pad = 40.0
    x1, y1 = max(0.0, float(np.min(xs)) - pad), max(0.0, float(np.min(ys)) - pad)
    x2, y2 = min(float(W), float(np.max(xs)) + pad), min(float(H), float(np.max(ys)) + pad)
    bbox = (x1, y1, x2, y2)

    # Crop and resize to 224x224
    base_ds = BaseHandObjectDataset(split="test", img_size=224, augment=False)
    crop_img = base_ds.crop_and_resize(img_rgb, bbox, target_size=224)
    cam_intr_updated = base_ds.update_intrinsics_for_crop_resize(cam_intr_raw, bbox, target_size=224)

    # Image tensor: (1, 3, 224, 224)
    img_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
    img_tensor = base_ds.normalize(img_tensor).unsqueeze(0).to(device)

    # Prepare batch tensors
    batch_cam_intr = torch.from_numpy(cam_intr_updated).float().unsqueeze(0).to(device)
    batch_hand_verts = verts.unsqueeze(0)
    batch_hand_joints = joints.unsqueeze(0)
    batch_palm = palm.unsqueeze(0)

    # 5. Load model and checkpoint
    cfg = load_config(args.config)
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[Infer] Running forward pass...")
    with torch.no_grad():
        preds = model(
            image=img_tensor,
            hand_verts=batch_hand_verts,
            hand_joints=batch_hand_joints,
            palm_coord=batch_palm,
            cam_intr=batch_cam_intr
        )

    # 6. Extract outputs
    to = preds["pred_trans"][0].cpu().numpy()                       # (3,)
    sparse_pts_local = preds["sparse_points"][0].cpu().numpy()       # (2048, 3)
    dense_pts_local = preds["dense_points"][0].cpu().numpy()         # (16384, 3)
    palm_np = palm.cpu().numpy()                                     # (3,)

    # Transform points to camera coordinates: P_cam = P_local + palm + to
    sparse_pts_cam = sparse_pts_local + palm_np + to
    dense_pts_cam = dense_pts_local + palm_np + to

    # 7. Save outputs
    # Point clouds (.ply)
    sparse_ply_path = os.path.join(args.output, "sparse_object.ply")
    dense_ply_path = os.path.join(args.output, "dense_object.ply")
    hand_ply_path = os.path.join(args.output, "predicted_hand.ply")

    export_point_cloud_ply(sparse_pts_cam, sparse_ply_path)
    export_point_cloud_ply(dense_pts_cam, dense_ply_path)
    export_point_cloud_ply(verts_np, hand_ply_path)

    # Optional surface mesh
    mesh_path = None
    if args.generate_mesh:
        mesh_path = os.path.join(args.output, "dense_object_mesh.ply")
        try:
            generate_mesh_from_point_cloud(dense_pts_cam, mesh_path)
            print(f"[Infer] Generated surface mesh: {mesh_path}")
        except Exception as e:
            print(f"[Infer] Mesh generation warning: {e}")

    # Summary JSON
    summary = {
        "hand_relative_translation_to": to.tolist(),
        "hand_palm_position": palm_np.tolist(),
        "object_centroid_cam": (palm_np + to).tolist(),
        "sparse_points_count": int(sparse_pts_cam.shape[0]),
        "dense_points_count": int(dense_pts_cam.shape[0]),
        "output_files": {
            "hand_mesh_ply": hand_ply_path,
            "sparse_object_ply": sparse_ply_path,
            "dense_object_ply": dense_ply_path,
            "reconstruction_mesh_ply": mesh_path
        }
    }
    summary_path = os.path.join(args.output, "reconstruction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print(f"[Infer] Inference complete! Outputs saved to {args.output}/:")
    print(f" - Predicted Hand:         {hand_ply_path}")
    print(f" - Sparse Cloud (2048):    {sparse_ply_path}")
    print(f" - Dense Cloud (16384):    {dense_ply_path}")
    if mesh_path and os.path.exists(mesh_path):
        print(f" - Reconstructed Mesh:     {mesh_path}")
    print(f" - Summary JSON:           {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

