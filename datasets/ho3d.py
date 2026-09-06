import os
import pickle
import numpy as np
import cv2
import trimesh
from .base import BaseHandObjectDataset, DummyHandObjectDataset


class HO3DDataset(BaseHandObjectDataset):
    """HO3D v3 Dataset adapter (Hampali et al., CVPR 2020).
    Expected structure under `data_root`:
        data_root/
            train/ (or split/)
                <seq_name>/
                    rgb/0000.jpg
                    meta/0000.pkl
            models/
                <obj_name>/textured_simple.obj
    """
    def __init__(
        self,
        data_root: str = "data/ho3d",
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
        self.models_dir = os.path.join(data_root, "models")

        # Gather sample index pairs (seq_name, frame_id)
        self.samples = []
        if os.path.isdir(self.split_dir):
            for seq in sorted(os.listdir(self.split_dir)):
                seq_path = os.path.join(self.split_dir, seq)
                rgb_dir = os.path.join(seq_path, "rgb")
                if os.path.isdir(rgb_dir):
                    for fname in sorted(os.listdir(rgb_dir)):
                        if fname.endswith((".jpg", ".png")):
                            frame_id = os.path.splitext(fname)[0]
                            self.samples.append((seq, frame_id))
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

        seq_name, frame_id = self.samples[idx]
        seq_path = os.path.join(self.split_dir, seq_name)

        # 1. Read Image
        img_path = os.path.join(seq_path, "rgb", f"{frame_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(seq_path, "rgb", f"{frame_id}.png")
        img_bgr = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        # 2. Read Meta
        meta_path = os.path.join(seq_path, "meta", f"{frame_id}.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        cam_mat = np.array(meta.get("camMat", meta.get("cam_intr", [
            [614.6, 0.0, 320.0],
            [0.0, 614.6, 240.0],
            [0.0, 0.0, 1.0]
        ])), dtype=np.float32)

        # 3. Hand vertices and joints
        hand_joints = np.array(meta.get("handJoints3D", np.zeros((21, 3))), dtype=np.float32)
        hand_verts = np.array(meta.get("handVerts3D", np.zeros((778, 3))), dtype=np.float32)
        palm_coord = np.array(meta.get("handTrans", hand_joints[0] if len(hand_joints) > 0 else np.zeros(3)), dtype=np.float32)

        # 4. Object translation & model
        obj_trans = np.array(meta.get("objTrans", np.zeros(3)), dtype=np.float32)
        # Translation relative to palm:
        rel_obj_trans = obj_trans - palm_coord

        obj_name = meta.get("objName", "")
        mesh_path = os.path.join(self.models_dir, obj_name, "textured_simple.obj")
        obj_mesh = None
        if os.path.exists(mesh_path):
            obj_mesh = trimesh.load(mesh_path, force="mesh")
            sparse_pts, dense_pts = self.sample_mesh_points(obj_mesh)
        else:
            sparse_pts = np.random.randn(self.num_sparse_points, 3).astype(np.float32) * 0.04
            dense_pts = np.random.randn(self.num_dense_points, 3).astype(np.float32) * 0.04

        # 5. 2D Bounding Box
        xs = (hand_verts[:, 0] / np.maximum(hand_verts[:, 2], 1e-4)) * cam_mat[0, 0] + cam_mat[0, 2]
        ys = (hand_verts[:, 1] / np.maximum(hand_verts[:, 2], 1e-4)) * cam_mat[1, 1] + cam_mat[1, 2]
        x1, y1 = max(0, float(np.min(xs)) - 30), max(0, float(np.min(ys)) - 30)
        x2, y2 = min(W, float(np.max(xs)) + 30), min(H, float(np.max(ys)) + 30)
        bbox = (x1, y1, x2, y2)

        return {
            "image_raw": img_raw,
            "bbox": bbox,
            "cam_intr_raw": cam_mat,
            "cam_extr": np.eye(4, dtype=np.float32),
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "palm_coord": palm_coord,
            "obj_trans": rel_obj_trans,
            "obj_points_sparse": sparse_pts,
            "obj_points_dense": dense_pts,
            "obj_mesh": obj_mesh,
            "meta": {"seq": seq_name, "frame_id": frame_id, "split": self.split}
        }

