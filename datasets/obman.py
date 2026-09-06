import os
import pickle
import numpy as np
import cv2
import trimesh
from .base import BaseHandObjectDataset, DummyHandObjectDataset


class ObManDataset(BaseHandObjectDataset):
    """ObMan dataset adapter (Hassao et al., CVPR 2019).
    Expected structure under `data_root`:
        data_root/
            train/ (or split/)
                rgb/
                    00000000.jpg
                meta/
                    00000000.pkl
            test/
                rgb/
                meta/
    """
    def __init__(
        self,
        data_root: str = "data/obman",
        split: str = "train",
        img_size: int = 224,
        num_sparse_points: int = 2048,
        num_dense_points: int = 16384,
        cache_dir: str | None = None,
        augment: bool = True,
        max_samples: int | None = None
    ):
        super().__init__(
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse_points,
            num_dense_points=num_dense_points,
            cache_dir=cache_dir,
            augment=augment
        )
        self.data_root = data_root
        self.split_dir = os.path.join(data_root, split)
        self.rgb_dir = os.path.join(self.split_dir, "rgb")
        self.meta_dir = os.path.join(self.split_dir, "meta")

        # Find sample files if present
        self.sample_ids = []
        if os.path.isdir(self.rgb_dir):
            all_files = sorted(os.listdir(self.rgb_dir))
            self.sample_ids = [os.path.splitext(f)[0] for f in all_files if f.endswith((".jpg", ".png"))]
            if max_samples is not None:
                self.sample_ids = self.sample_ids[:max_samples]

        # If data directory doesn't exist yet, enable mock fallback for seamless testing
        self.is_mock = len(self.sample_ids) == 0
        if self.is_mock:
            self._dummy = DummyHandObjectDataset(
                length=32 if max_samples is None else max_samples,
                split=split,
                img_size=img_size,
                num_sparse_points=num_sparse_points,
                num_dense_points=num_dense_points,
                augment=self.augment
            )

    def __len__(self) -> int:
        if self.is_mock:
            return len(self._dummy)
        return len(self.sample_ids)

    def get_raw_sample(self, idx: int) -> dict:
        if self.is_mock:
            return self._dummy.get_raw_sample(idx)

        sample_id = self.sample_ids[idx]

        # 1. Read image
        img_path = os.path.join(self.rgb_dir, f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.rgb_dir, f"{sample_id}.png")
        img_bgr = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        # 2. Read metadata
        meta_path = os.path.join(self.meta_dir, f"{sample_id}.pkl")
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)

        # 3. Extract camera intrinsics
        cam_intr = meta_data.get("cam_intr", meta_data.get("cam_calib", None))
        if cam_intr is None:
            # Construct default if missing
            cam_intr = np.array([
                [480.0, 0.0, W / 2.0],
                [0.0, 480.0, H / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            cam_intr = np.array(cam_intr, dtype=np.float32)

        # 4. Extract hand mesh / joints
        hand_verts = np.array(meta_data.get("hand_verts", meta_data.get("verts", np.zeros((778, 3)))), dtype=np.float32)
        hand_joints = np.array(meta_data.get("hand_joints", meta_data.get("joints", np.zeros((21, 3)))), dtype=np.float32)
        palm_coord = np.array(meta_data.get("palm_coord", hand_joints[0] if len(hand_joints) > 0 else np.zeros(3)), dtype=np.float32)

        # 5. Extract bounding box
        bbox = meta_data.get("bbox", None)
        if bbox is None:
            # Compute 2D bounding box from projected hand & object points
            xs = (hand_verts[:, 0] / np.maximum(hand_verts[:, 2], 1e-4)) * cam_intr[0, 0] + cam_intr[0, 2]
            ys = (hand_verts[:, 1] / np.maximum(hand_verts[:, 2], 1e-4)) * cam_intr[1, 1] + cam_intr[1, 2]
            x1, y1 = max(0, float(np.min(xs)) - 30), max(0, float(np.min(ys)) - 30)
            x2, y2 = min(W, float(np.max(xs)) + 30), min(H, float(np.max(ys)) + 30)
            bbox = (x1, y1, x2, y2)
        else:
            bbox = tuple(float(x) for x in bbox)

        # 6. Extract object translation and points
        obj_trans = np.array(meta_data.get("obj_trans", np.zeros(3)), dtype=np.float32)
        sparse_pts = meta_data.get("obj_points_sparse", None)
        dense_pts = meta_data.get("obj_points_dense", None)

        obj_mesh = None
        if sparse_pts is None or dense_pts is None:
            mesh_path = meta_data.get("obj_mesh_path", None)
            if mesh_path and os.path.exists(mesh_path):
                obj_mesh = trimesh.load(mesh_path, force="mesh")
            else:
                # Sample sphere placeholder if mesh file is not linked
                sparse_pts = np.random.randn(self.num_sparse_points, 3).astype(np.float32) * 0.05
                dense_pts = np.random.randn(self.num_dense_points, 3).astype(np.float32) * 0.05

        return {
            "image_raw": img_raw,
            "bbox": bbox,
            "cam_intr_raw": cam_intr,
            "cam_extr": np.eye(4, dtype=np.float32),
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "palm_coord": palm_coord,
            "obj_trans": obj_trans,
            "obj_points_sparse": sparse_pts,
            "obj_points_dense": dense_pts,
            "obj_mesh": obj_mesh,
            "meta": {"sample_id": sample_id, "split": self.split}
        }

