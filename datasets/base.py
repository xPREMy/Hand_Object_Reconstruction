import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import trimesh


class BaseHandObjectDataset(Dataset):
    """Abstract Base Dataset interface for Hand-Object Reconstruction.
    Produces dictionary containing:
        - image: (3, 224, 224) torch.FloatTensor normalized
        - bbox: (4,) np.ndarray [x1, y1, x2, y2]
        - cam_intr: (3, 3) torch.FloatTensor updated for crop and resize
        - cam_extr: (4, 4) torch.FloatTensor
        - hand_verts: (778, 3) torch.FloatTensor in camera frame
        - hand_joints: (21, 3) torch.FloatTensor in camera frame
        - palm_coord: (3,) torch.FloatTensor in camera frame
        - gt_trans: (3,) torch.FloatTensor (object centroid - hand palm)
        - gt_points_sparse: (2048, 3) torch.FloatTensor (centered at object centroid)
        - gt_points_dense: (16384, 3) torch.FloatTensor (centered at object centroid)
        - meta: dict with sample identifiers
    """
    def __init__(
        self,
        split: str = "train",
        img_size: int = 224,
        num_sparse_points: int = 2048,
        num_dense_points: int = 16384,
        cache_dir: str | None = None,
        augment: bool = True
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.num_sparse_points = num_sparse_points
        self.num_dense_points = num_dense_points
        self.cache_dir = cache_dir
        self.augment = augment and (split == "train")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Image normalization (ImageNet stats)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def get_raw_sample(self, idx: int) -> dict:
        """Returns raw data before crop/resize/augmentation:
            - image_raw: (H, W, 3) uint8 numpy array
            - bbox: [x1, y1, x2, y2]
            - cam_intr_raw: (3, 3) float32
            - cam_extr: (4, 4) float32
            - hand_verts: (778, 3) float32
            - hand_joints: (16, 3) or (21, 3) float32
            - palm_coord: (3,) float32
            - obj_mesh: trimesh.Trimesh or None
            - obj_points_sparse: (2048, 3) if precomputed
            - obj_points_dense: (16384, 3) if precomputed
            - obj_trans: (3,) float32
            - meta: dict
        """
        raise NotImplementedError

    def sample_mesh_points(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly samples sparse and dense points on mesh surface."""
        dense_pts, _ = trimesh.sample.sample_surface(mesh, self.num_dense_points)
        # Select first num_sparse_points from dense sample or resample
        sparse_pts = dense_pts[:self.num_sparse_points].copy()
        return sparse_pts.astype(np.float32), dense_pts.astype(np.float32)

    def update_intrinsics_for_crop_resize(
        self,
        cam_intr: np.ndarray,
        bbox: tuple[float, float, float, float],
        target_size: int = 224
    ) -> np.ndarray:
        """Correctly updates camera intrinsics after cropping bbox [x1, y1, x2, y2]
        and resizing to (target_size, target_size).
        """
        x1, y1, x2, y2 = bbox
        w_crop = max(x2 - x1, 1e-4)
        h_crop = max(y2 - y1, 1e-4)

        sx = float(target_size) / float(w_crop)
        sy = float(target_size) / float(h_crop)

        new_k = cam_intr.copy()
        new_k[0, 0] = cam_intr[0, 0] * sx
        new_k[1, 1] = cam_intr[1, 1] * sy
        new_k[0, 2] = (cam_intr[0, 2] - x1) * sx
        new_k[1, 2] = (cam_intr[1, 2] - y1) * sy
        return new_k

    def crop_and_resize(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        target_size: int = 224
    ) -> np.ndarray:
        """Crops bbox with boundary padding if necessary, then resizes to target_size."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Clamp crop coordinates and pad if bbox exceeds image bounds
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            x1 += pad_left
            x2 += pad_left
            y1 += pad_top
            y2 += pad_top

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        else:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def __getitem__(self, idx: int) -> dict:
        raw = self.get_raw_sample(idx)

        img_raw = raw["image_raw"]  # (H, W, 3) uint8 RGB
        x1, y1, x2, y2 = raw["bbox"]
        cam_intr = raw["cam_intr_raw"].copy()
        cam_extr = raw.get("cam_extr", np.eye(4, dtype=np.float32))

        # Data augmentation (Section 4.3):
        # random rotation [-90, 90], random resize [0.8, 1.2], random bbox shift, color jitter
        if self.augment:
            # 1. Bbox shift
            bw = x2 - x1
            bh = y2 - y1
            shift_x = (random.random() - 0.5) * 0.1 * bw
            shift_y = (random.random() - 0.5) * 0.1 * bh
            x1 += shift_x
            x2 += shift_x
            y1 += shift_y
            y2 += shift_y

            # 2. Random scale [0.8, 1.2]
            scale = random.uniform(0.8, 1.2)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = bw * scale
            bh = bh * scale
            x1 = cx - bw / 2.0
            x2 = cx + bw / 2.0
            y1 = cy - bh / 2.0
            y2 = cy + bh / 2.0

        bbox = (x1, y1, x2, y2)

        # Crop and resize image
        crop_img = self.crop_and_resize(img_raw, bbox, self.img_size)

        # Update intrinsics for crop and resize
        updated_intr = self.update_intrinsics_for_crop_resize(cam_intr, bbox, self.img_size)

        # Convert image to tensor (C, H, W) normalized in [0, 1]
        img_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0

        # Color jitter augmentation
        if self.augment:
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            img_tensor = color_jitter(img_tensor)

        img_tensor = self.normalize(img_tensor)

        # Get point clouds
        if "obj_points_sparse" in raw and "obj_points_dense" in raw:
            sparse_pts = raw["obj_points_sparse"]
            dense_pts = raw["obj_points_dense"]
        elif raw.get("obj_mesh") is not None:
            sparse_pts, dense_pts = self.sample_mesh_points(raw["obj_mesh"])
        else:
            # Fallback dummy points if mesh not provided
            sparse_pts = np.zeros((self.num_sparse_points, 3), dtype=np.float32)
            dense_pts = np.zeros((self.num_dense_points, 3), dtype=np.float32)

        # Assemble output dictionary
        return {
            "image": img_tensor,
            "bbox": np.array(bbox, dtype=np.float32),
            "cam_intr": torch.from_numpy(updated_intr).float(),
            "cam_extr": torch.from_numpy(cam_extr).float(),
            "hand_verts": torch.from_numpy(raw["hand_verts"]).float(),
            "hand_joints": torch.from_numpy(raw["hand_joints"]).float(),
            "palm_coord": torch.from_numpy(raw["palm_coord"]).float(),
            "gt_trans": torch.from_numpy(raw["obj_trans"]).float(),
            "gt_points_sparse": torch.from_numpy(sparse_pts).float(),
            "gt_points_dense": torch.from_numpy(dense_pts).float(),
            "meta": raw.get("meta", {"idx": idx})
        }


class DummyHandObjectDataset(BaseHandObjectDataset):
    """Synthetic dummy dataset for testing, debugging, and verification."""
    def __init__(
        self,
        length: int = 16,
        split: str = "train",
        img_size: int = 224,
        num_sparse_points: int = 2048,
        num_dense_points: int = 16384,
        augment: bool = False
    ):
        super().__init__(
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse_points,
            num_dense_points=num_dense_points,
            augment=augment
        )
        self.length = length

    def __len__(self) -> int:
        return self.length

    def get_raw_sample(self, idx: int) -> dict:
        np.random.seed(idx)

        # Synthetic image (H, W, 3)
        img_raw = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        # Synthetic bbox around hand-object
        bbox = (150.0, 100.0, 450.0, 400.0)

        # Synthetic intrinsics
        cam_intr_raw = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Synthetic hand in camera space (depth ~ 0.6m)
        palm = np.array([0.02, 0.05, 0.6], dtype=np.float32)
        hand_verts = np.random.randn(778, 3).astype(np.float32) * 0.05 + palm
        hand_joints = np.random.randn(21, 3).astype(np.float32) * 0.04 + palm

        # Synthetic object (centroid relative to palm)
        obj_trans = np.array([0.01, -0.02, 0.05], dtype=np.float32)

        # Synthetic sphere point cloud for object centered at origin
        theta = np.random.uniform(0, 2 * math.pi, self.num_dense_points)
        phi = np.random.uniform(0, math.pi, self.num_dense_points)
        r = 0.04  # 4 cm radius
        xs = r * np.sin(phi) * np.cos(theta)
        ys = r * np.sin(phi) * np.sin(theta)
        zs = r * np.cos(phi)
        dense_pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
        sparse_pts = dense_pts[:self.num_sparse_points].copy()

        return {
            "image_raw": img_raw,
            "bbox": bbox,
            "cam_intr_raw": cam_intr_raw,
            "cam_extr": np.eye(4, dtype=np.float32),
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "palm_coord": palm,
            "obj_trans": obj_trans,
            "obj_points_sparse": sparse_pts,
            "obj_points_dense": dense_pts,
            "meta": {"sample_id": f"dummy_{idx}", "split": self.split}
        }

