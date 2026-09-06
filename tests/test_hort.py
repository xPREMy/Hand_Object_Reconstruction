import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import cv2

from models.image_encoder import ImageEncoder
from models.hand_encoder import HandEncoder
from models.sparse_decoder import SparseDecoder
from models.dense_decoder import DenseDecoder
from models.hort import HORT
from losses.chamfer import ChamferLoss, HORTLoss, pairwise_distance_squared, chamfer_distance_chunked
from datasets.base import DummyHandObjectDataset, BaseHandObjectDataset


class TestHORT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_1_image_encoder_shapes_and_freezing(self):
        """1. Test image encoder output shape and layer freezing."""
        encoder = ImageEncoder(pretrained=False).to(self.device)
        x = torch.randn(2, 3, 224, 224, device=self.device)
        feat = encoder(x)
        self.assertEqual(feat.shape, (2, 257, 1024))

        # Check that early blocks are frozen (requires_grad = False)
        for i in range(12):
            for p in encoder.backbone.blocks[i].parameters():
                self.assertFalse(p.requires_grad)

        # Check that later blocks are trainable (requires_grad = True)
        for i in range(12, 24):
            for p in encoder.backbone.blocks[i].parameters():
                self.assertTrue(p.requires_grad)

    def test_2_hand_encoder_shapes(self):
        """2. Test hand encoder 22-coordinate extraction and PointNet output."""
        encoder = HandEncoder().to(self.device)
        verts = torch.randn(2, 778, 3, device=self.device)
        joints = torch.randn(2, 21, 3, device=self.device)
        palm = torch.randn(2, 3, device=self.device)

        fh = encoder(verts, joints, palm)
        self.assertEqual(fh.shape, (2, 1024))

        # Also test with 16 joints
        joints_16 = torch.randn(2, 16, 3, device=self.device)
        fh_16 = encoder(verts, joints_16, palm)
        self.assertEqual(fh_16.shape, (2, 1024))

    def test_3_sparse_decoder_shapes(self):
        """3. Test sparse decoder joint prediction of pose and 2048 points."""
        decoder = SparseDecoder().to(self.device)
        fv = torch.randn(2, 257, 1024, device=self.device)
        fh = torch.randn(2, 1024, device=self.device)

        to, p_s = decoder(fv, fh)
        self.assertEqual(to.shape, (2, 3))
        self.assertEqual(p_s.shape, (2, 2048, 3))

    def test_4_dense_decoder_shapes(self):
        """4. Test dense decoder projection, kNN self-attention, and upsampling to 16,384 points."""
        decoder = DenseDecoder().to(self.device)
        p_s = torch.randn(2, 2048, 3, device=self.device)
        to = torch.randn(2, 3, device=self.device)
        palm = torch.tensor([[0.0, 0.0, 0.6], [0.0, 0.0, 0.6]], device=self.device)
        fv = torch.randn(2, 257, 1024, device=self.device)
        verts = torch.randn(2, 778, 3, device=self.device) + palm.unsqueeze(1)
        cam_intr = torch.tensor([
            [[500.0, 0.0, 112.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]],
            [[500.0, 0.0, 112.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]]
        ], device=self.device)

        p_dense = decoder(
            p_s=p_s,
            to=to,
            palm_coord=palm,
            fv=fv,
            hand_verts=verts,
            cam_intr=cam_intr
        )
        self.assertEqual(p_dense.shape, (2, 16384, 3))

    def test_5_perspective_projection(self):
        """5. Test perspective projection math and bilinear feature sampling."""
        decoder = DenseDecoder().to(self.device)
        feat_map = torch.ones(1, 128, 16, 16, device=self.device) * 5.0
        p_cam = torch.tensor([[[0.0, 0.0, 1.0]]], device=self.device)  # Center point at Z=1
        cam_intr = torch.tensor([[[224.0, 0.0, 112.0], [0.0, 224.0, 112.0], [0.0, 0.0, 1.0]]], device=self.device)

        sampled = decoder._project_and_sample_features(p_cam, feat_map, cam_intr)
        self.assertEqual(sampled.shape, (1, 1, 128))
        self.assertTrue(torch.allclose(sampled, torch.full_like(sampled, 5.0), atol=1e-3))

    def test_6_chamfer_loss(self):
        """6. Test Chamfer loss values and backward gradients."""
        p1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
        p2 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)

        # Identical point clouds -> CD = 0.0
        cd_zero = chamfer_distance_chunked(p1, p2)
        self.assertAlmostEqual(cd_zero.item(), 0.0, places=5)

        # Shifted point cloud
        p3 = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=torch.float32)
        cd_shift = chamfer_distance_chunked(p1, p3)
        # min dist squared for both points is 1.0 -> mean is 1.0 -> sum of directions is 2.0
        self.assertAlmostEqual(cd_shift.item(), 2.0, places=4)

        # Backward pass gradient check
        cd_shift.backward()
        self.assertIsNotNone(p1.grad)
        self.assertEqual(p1.grad.shape, p1.shape)

    def test_7_dataset_output_contract(self):
        """7. Test dataset item shapes and dictionary contract."""
        dataset = DummyHandObjectDataset(length=4)
        sample = dataset[0]

        expected_keys = [
            "image", "bbox", "cam_intr", "cam_extr", "hand_verts",
            "hand_joints", "palm_coord", "gt_trans", "gt_points_sparse",
            "gt_points_dense", "meta"
        ]
        for key in expected_keys:
            self.assertIn(key, sample)

        self.assertEqual(sample["image"].shape, (3, 224, 224))
        self.assertEqual(sample["cam_intr"].shape, (3, 3))
        self.assertEqual(sample["hand_verts"].shape, (778, 3))
        self.assertEqual(sample["hand_joints"].shape, (21, 3))
        self.assertEqual(sample["palm_coord"].shape, (3,))
        self.assertEqual(sample["gt_trans"].shape, (3,))
        self.assertEqual(sample["gt_points_sparse"].shape, (2048, 3))
        self.assertEqual(sample["gt_points_dense"].shape, (16384, 3))

    def test_8_full_forward_and_backward_pass(self):
        """8. Test full HORT forward pass, loss calculation, backward pass, and optimizer step."""
        model = HORT(pretrained=False).to(self.device)
        criterion = HORTLoss()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        dataset = DummyHandObjectDataset(length=2)
        batch = {k: torch.stack([dataset[0][k], dataset[1][k]]).to(self.device)
                 for k in ["image", "hand_verts", "hand_joints", "palm_coord", "cam_intr",
                           "gt_trans", "gt_points_sparse", "gt_points_dense"]}

        optimizer.zero_grad()
        preds = model(
            image=batch["image"],
            hand_verts=batch["hand_verts"],
            hand_joints=batch["hand_joints"],
            palm_coord=batch["palm_coord"],
            cam_intr=batch["cam_intr"]
        )

        self.assertEqual(preds["pred_trans"].shape, (2, 3))
        self.assertEqual(preds["sparse_points"].shape, (2, 2048, 3))
        self.assertEqual(preds["dense_points"].shape, (2, 16384, 3))

        loss_dict = criterion(
            pred_trans=preds["pred_trans"],
            gt_trans=batch["gt_trans"],
            pred_sparse_points=preds["sparse_points"],
            gt_sparse_points=batch["gt_points_sparse"],
            pred_dense_points=preds["dense_points"],
            gt_dense_points=batch["gt_points_dense"]
        )

        loss = loss_dict["loss"]
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

        # Check trainable parameters received gradients
        has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0
                       for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad)

        optimizer.step()

    def test_9_checkpoint_save_and_reload(self):
        """9. Test checkpoint serialization and restoration."""
        tmp_dir = tempfile.mkdtemp()
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")

        try:
            model1 = HORT(pretrained=False).to(self.device)
            state_dict = model1.state_dict()
            torch.save({"model_state_dict": state_dict, "epoch": 1, "best_loss": 0.5}, ckpt_path)

            model2 = HORT(pretrained=False).to(self.device)
            loaded = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model2.load_state_dict(loaded["model_state_dict"])

            # Verify parameters match exactly
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()

