import os
import argparse
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt


def render_to_image(
    hand_ply_path: str,
    obj_ply_path: str,
    output_image_path: str = "visualization.png"
):
    """Renders 3D hand and object point cloud into a 2D multi-angle PNG visualization."""
    fig = plt.figure(figsize=(12, 6))

    # Load geometries
    pcd_hand = trimesh.load(hand_ply_path)
    pcd_obj = trimesh.load(obj_ply_path)

    hand_pts = np.array(pcd_hand.vertices if hasattr(pcd_hand, "vertices") else pcd_hand.points)
    obj_pts = np.array(pcd_obj.vertices if hasattr(pcd_obj, "vertices") else pcd_obj.points)

    angles = [(20, 45), (0, 0), (90, 0)]
    titles = ["Perspective View (45°)", "Front View (X-Y)", "Top View (X-Z)"]

    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        # Plot hand points (blue)
        ax.scatter(hand_pts[:, 0], hand_pts[:, 1], hand_pts[:, 2], c="royalblue", s=2, label="Hand", alpha=0.6)
        # Plot object points (crimson)
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2], c="crimson", s=4, label="Reconstructed Object")

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        if i == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()
    print(f"[Visualize] Rendered multi-view visualization image to: {output_image_path}")


def show_interactive(hand_ply_path: str, obj_ply_path: str, mesh_path: str | None = None):
    """Opens an interactive 3D window with Open3D."""
    geometries = []

    if os.path.exists(hand_ply_path):
        pcd_hand = o3d.io.read_point_cloud(hand_ply_path)
        pcd_hand.paint_uniform_color([0.2, 0.4, 0.9])  # Blue for hand
        geometries.append(pcd_hand)

    if mesh_path and os.path.exists(mesh_path):
        mesh_obj = o3d.io.read_triangle_mesh(mesh_path)
        mesh_obj.paint_uniform_color([0.9, 0.2, 0.2])  # Red for object
        mesh_obj.compute_vertex_normals()
        geometries.append(mesh_obj)
    elif os.path.exists(obj_ply_path):
        pcd_obj = o3d.io.read_point_cloud(obj_ply_path)
        pcd_obj.paint_uniform_color([0.9, 0.2, 0.2])  # Red for object
        geometries.append(pcd_obj)

    if not geometries:
        print("[Visualize] No geometries found to display.")
        return

    print("[Visualize] Opening interactive 3D window... (Press 'Q' or close window to exit)")
    o3d.visualization.draw_geometries(geometries, window_name="HORT 3D Reconstruction")


def main():
    parser = argparse.ArgumentParser(description="Visualize HORT 3D Reconstruction")
    parser.add_argument("--output_dir", type=str, default="output_hort", help="Directory containing infer.py outputs")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive Open3D window")
    parser.add_argument("--save_render", type=str, default="reconstruction_render.png", help="Path to save 2D multi-angle render")
    args = parser.parse_args()

    hand_ply = os.path.join(args.output_dir, "predicted_hand.ply")
    obj_dense_ply = os.path.join(args.output_dir, "dense_object.ply")
    obj_mesh_ply = os.path.join(args.output_dir, "dense_object_mesh.ply")

    if not os.path.exists(obj_dense_ply):
        print(f"[Visualize] Error: {obj_dense_ply} does not exist. Run infer.py first.")
        return

    # 1. Save 2D multi-angle PNG render (viewable anywhere without GUI window)
    render_to_image(hand_ply, obj_dense_ply, args.save_render)

    # 2. Interactive window if requested
    if args.interactive:
        show_interactive(hand_ply, obj_dense_ply, obj_mesh_ply if os.path.exists(obj_mesh_ply) else None)


if __name__ == "__main__":
    main()

