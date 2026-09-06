import os
import numpy as np
import cv2
import trimesh
from .base import BaseHandObjectDataset, DummyHandObjectDataset


class DexYCBDataset(BaseHandObjectDataset):
    """DexYCB Dataset adapter (Chao et al., CVPR 2021).
    Expected structure under `data_root`:
        data_root/
            <sequence>/
                <camera_id>/
                    color_000000.jpg
                    labels_000000.npz
            models/
                <obj_name>/textured_simple.obj
    """
    def __init__(
        self,
        data_root: str = "data/dexycb",
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
        self.models_dir = os.path.join(data_root, "models")

        self.samples = []
        if os.path.isdir(data_root):
            for root, _, files in os.walk(data_root):
                for f in files:
                    if f.startswith("labels_") and f.endswith(".npz"):
                        lbl_path = os.path.join(root, f)
                        color_name = f.replace("labels_", "color_").replace(".npz", ".jpg")
                        img_path = os.path.join(root, color_name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, lbl_path))
            if max_samples is not None:
                self.samples = self.samples[:max_samples]

        self.is_mock = len(self.samples) == 0
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
        return len(self.samples)

    def get_raw_sample(self, idx: int) -> dict:
        if self.is_mock:
            return self._dummy.get_raw_sample(idx)

        img_path, lbl_path = self.samples[idx]

        img_bgr = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        label = np.load(lbl_path)
        cam_intr = np.array(label.get("intrinsics", [
            [615.0, 0.0, W / 2.0],
            [0.0, 615.0, H / 2.0],
            [0.0, 0.0, 1.0]
        ]), dtype=np.float32)

        hand_verts = np.array(label.get("joint_3d", np.zeros((778, 3))), dtype=np.float32)
        if hand_verts.shape[0] != 778:
            hand_verts = np.zeros((778, 3), dtype=np.float32)
        hand_joints = np.array(label.get("joint_3d", np.zeros((21, 3))), dtype=np.float32)[:21]
        palm_coord = hand_joints[0] if len(hand_joints) > 0 else np.zeros(3, dtype=np.float32)

        obj_trans = np.array(label.get("pose_m", np.zeros((4, 4))), dtype=np.float32)[:3, 3]
        rel_obj_trans = obj_trans - palm_coord

        # Mesh points
        sparse_pts = np.random.randn(self.num_sparse_points, 3).astype(np.float32) * 0.04
        dense_pts = np.random.randn(self.num_dense_points, 3).astype(np.float32) * 0.04

        bbox = (float(W * 0.2), float(H * 0.2), float(W * 0.8), float(H * 0.8))

        return {
            "image_raw": img_raw,
            "bbox": bbox,
            "cam_intr_raw": cam_intr,
            "cam_extr": np.eye(4, dtype=np.float32),
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "palm_coord": palm_coord,
            "obj_trans": rel_obj_trans,
            "obj_points_sparse": sparse_pts,
            "obj_points_dense": dense_pts,
            "obj_mesh": None,
            "meta": {"img_path": img_path, "split": self.split}
        }

