import os
import json
import numpy as np
import cv2
import trimesh
from .base import BaseHandObjectDataset, DummyHandObjectDataset


class MOWDataset(BaseHandObjectDataset):
    """MOW Dataset adapter (Shan et al., CVPR 2020 / 100 Days of Hands).
    Expected structure under `data_root`:
        data_root/
            images/
                <img_id>.jpg
            annotations/
                train.json (or <split>.json)
    """
    def __init__(
        self,
        data_root: str = "data/mow",
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
        self.images_dir = os.path.join(data_root, "images")
        self.ann_path = os.path.join(data_root, "annotations", f"{split}.json")

        self.samples = []
        if os.path.isfile(self.ann_path):
            with open(self.ann_path, "r") as f:
                self.samples = json.load(f)
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

        item = self.samples[idx]
        img_path = os.path.join(self.images_dir, item["image_file"])

        img_bgr = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        cam_intr = np.array(item.get("cam_intr", [
            [500.0, 0.0, W / 2.0],
            [0.0, 500.0, H / 2.0],
            [0.0, 0.0, 1.0]
        ]), dtype=np.float32)

        hand_verts = np.array(item.get("hand_verts", np.zeros((778, 3))), dtype=np.float32)
        hand_joints = np.array(item.get("hand_joints", np.zeros((21, 3))), dtype=np.float32)
        palm_coord = np.array(item.get("palm_coord", hand_joints[0] if len(hand_joints) > 0 else np.zeros(3)), dtype=np.float32)

        obj_trans = np.array(item.get("obj_trans", np.zeros(3)), dtype=np.float32)
        rel_obj_trans = obj_trans - palm_coord

        sparse_pts = np.random.randn(self.num_sparse_points, 3).astype(np.float32) * 0.04
        dense_pts = np.random.randn(self.num_dense_points, 3).astype(np.float32) * 0.04

        bbox = tuple(item.get("bbox", (float(W * 0.2), float(H * 0.2), float(W * 0.8), float(H * 0.8))))

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

