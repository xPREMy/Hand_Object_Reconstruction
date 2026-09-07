import os
import numpy as np
import cv2
import trimesh
from .base import BaseHandObjectDataset


class MOWDataset(BaseHandObjectDataset):
    """MOW adapter for the released image/OBJ pairs.
    Expected structure under `data_root`:
        data_root/
            images/
                <sample_id>.jpg
            models/
                <sample_id>.obj

    This download has no hand, camera, or object-pose annotations. The adapter
    therefore uses a canonical hand proxy, a default camera, and object-centred
    targets. This makes the training pipeline runnable, but the resulting model
    is not a replacement for training on metric hand/pose annotations.
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

        image_stems = {
            os.path.splitext(name)[0]
            for name in os.listdir(self.images_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        } if os.path.isdir(self.images_dir) else set()
        models_dir = os.path.join(data_root, "models")
        model_stems = {
            os.path.splitext(name)[0]
            for name in os.listdir(models_dir)
            if name.lower().endswith(".obj")
        } if os.path.isdir(models_dir) else set()
        sample_ids = sorted(image_stems & model_stems)
        if not sample_ids:
            raise FileNotFoundError(
                f"No matching MOW image/OBJ pairs found under {data_root!r}. "
                "Expected images/*.jpg and models/*.obj."
            )

        # Keep every run reproducible and ensure train/test have no overlap.
        n = len(sample_ids)
        train_end = max(1, int(0.8 * n))
        val_end = max(train_end + 1, int(0.9 * n))
        if split == "train":
            sample_ids = sample_ids[:train_end]
        elif split in ("val", "valid", "validation"):
            sample_ids = sample_ids[train_end:val_end]
        elif split == "test":
            sample_ids = sample_ids[val_end:]
        else:
            raise ValueError(f"Unknown MOW split {split!r}; use train, val, or test")

        self.samples = sample_ids if max_samples is None else sample_ids[:max_samples]
        if not self.samples:
            raise ValueError(f"MOW split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw_sample(self, idx: int) -> dict:
        sample_id = self.samples[idx]
        img_path = next(
            os.path.join(self.images_dir, f"{sample_id}{ext}")
            for ext in (".jpg", ".jpeg", ".png")
            if os.path.exists(os.path.join(self.images_dir, f"{sample_id}{ext}"))
        )
        mesh_path = os.path.join(self.data_root, "models", f"{sample_id}.obj")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read MOW image: {img_path}")
        img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        cam_intr = np.array([
            [float(max(H, W)), 0.0, W / 2.0],
            [0.0, float(max(H, W)), H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Canonical proxy required by HORT when the download has no hand labels.
        rng = np.random.default_rng(idx)
        palm_coord = np.array([0.0, 0.0, 0.6], dtype=np.float32)
        hand_verts = (palm_coord + rng.normal(0.0, 0.045, (778, 3))).astype(np.float32)
        hand_joints = (palm_coord + rng.normal(0.0, 0.035, (21, 3))).astype(np.float32)

        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            raise ValueError(f"Invalid MOW OBJ mesh: {mesh_path}")
        bounds = mesh.bounds
        if float(np.max(bounds[1] - bounds[0])) > 2.0:
            mesh.apply_scale(0.01)
        mesh.apply_translation(-mesh.centroid)
        sparse_pts, dense_pts = self.sample_mesh_points(mesh)

        bbox = (float(W * 0.1), float(H * 0.1), float(W * 0.9), float(H * 0.9))

        return {
            "image_raw": img_raw,
            "bbox": bbox,
            "cam_intr_raw": cam_intr,
            "cam_extr": np.eye(4, dtype=np.float32),
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "palm_coord": palm_coord,
            "obj_trans": np.zeros(3, dtype=np.float32),
            "obj_points_sparse": sparse_pts,
            "obj_points_dense": dense_pts,
            "obj_mesh": mesh,
            "meta": {"sample_id": sample_id, "img_path": img_path, "split": self.split}
        }

