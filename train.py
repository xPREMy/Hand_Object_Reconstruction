import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from models.hort import HORT
from losses.chamfer import HORTLoss
from datasets.base import DummyHandObjectDataset
from datasets.obman import ObManDataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, dataset_name: str, split: str = "train", max_samples: int | None = None):
    data_cfg = cfg.get("dataset", {})
    img_size = data_cfg.get("image_size", 224)
    num_sparse = data_cfg.get("num_sparse_points", 2048)
    num_dense = data_cfg.get("num_dense_points", 16384)

    if dataset_name == "dummy":
        return DummyHandObjectDataset(
            length=16 if max_samples is None else max_samples,
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse,
            num_dense_points=num_dense,
            augment=(split == "train")
        )
    elif dataset_name == "obman":
        data_root = data_cfg.get("data_root", "data/obman")
        return ObManDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse,
            num_dense_points=num_dense,
            augment=(split == "train"),
            max_samples=max_samples
        )
    elif dataset_name == "ho3d":
        from datasets.ho3d import HO3DDataset
        data_root = data_cfg.get("data_root", "data/ho3d")
        return HO3DDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse,
            num_dense_points=num_dense,
            augment=(split == "train"),
            max_samples=max_samples
        )
    elif dataset_name == "dexycb":
        from datasets.dexycb import DexYCBDataset
        data_root = data_cfg.get("data_root", "data/dexycb")
        return DexYCBDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse,
            num_dense_points=num_dense,
            augment=(split == "train"),
            max_samples=max_samples
        )
    elif dataset_name == "mow":
        from datasets.mow import MOWDataset
        data_root = data_cfg.get("data_root", "data/mow")
        return MOWDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            num_sparse_points=num_sparse,
            num_dense_points=num_dense,
            augment=(split == "train"),
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(cfg: dict, device: torch.device) -> HORT:
    m_cfg = cfg.get("model", {})
    img_cfg = m_cfg.get("image_encoder", {})
    hand_cfg = m_cfg.get("hand_encoder", {})
    sp_cfg = m_cfg.get("sparse_decoder", {})
    de_cfg = m_cfg.get("dense_decoder", {})

    model = HORT(
        backbone_name=img_cfg.get("backbone", "vit_large_patch14_dinov2"),
        pretrained=img_cfg.get("pretrained", True),
        img_size=img_cfg.get("img_size", 224),
        embed_dim=img_cfg.get("embed_dim", 1024),
        num_frozen_layers=img_cfg.get("num_frozen_layers", 12),
        num_hand_verts=hand_cfg.get("num_verts", 778),
        num_coords=hand_cfg.get("num_coords", 22),
        hand_in_dim=hand_cfg.get("in_dim", 67),
        hand_feat_dim=hand_cfg.get("out_dim", 1024),
        sparse_hidden_dim=sp_cfg.get("hidden_dim", 512),
        sparse_num_layers=sp_cfg.get("num_layers", 10),
        sparse_num_heads=sp_cfg.get("num_heads", 8),
        sparse_ffn_dim=sp_cfg.get("ffn_dim", 2048),
        num_sparse_points=sp_cfg.get("num_sparse_points", 2048),
        dense_feat_map_dim=de_cfg.get("feature_map_dim", 128),
        dense_k_knn=de_cfg.get("k_knn", 16),
        first_upsample_factor=de_cfg.get("first_upsample_factor", 2),
        second_upsample_factor=de_cfg.get("second_upsample_factor", 4),
        num_dense_points=de_cfg.get("num_dense_points", 16384)
    )
    return model.to(device)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def train_one_epoch(
    model: HORT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: HORTLoss,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    log_interval: int = 10
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None and device.type == "cuda"

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for step, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        hand_verts = batch["hand_verts"].to(device, non_blocking=True)
        hand_joints = batch["hand_joints"].to(device, non_blocking=True)
        palm_coord = batch["palm_coord"].to(device, non_blocking=True)
        cam_intr = batch["cam_intr"].to(device, non_blocking=True)
        gt_trans = batch["gt_trans"].to(device, non_blocking=True)
        gt_sparse = batch["gt_points_sparse"].to(device, non_blocking=True)
        gt_dense = batch["gt_points_dense"].to(device, non_blocking=True)

        # Forward pass with Automatic Mixed Precision (AMP)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            preds = model(
                image=images,
                hand_verts=hand_verts,
                hand_joints=hand_joints,
                palm_coord=palm_coord,
                cam_intr=cam_intr
            )

            loss_dict = criterion(
                pred_trans=preds["pred_trans"],
                gt_trans=gt_trans,
                pred_sparse_points=preds["sparse_points"],
                gt_sparse_points=gt_sparse,
                pred_dense_points=preds["dense_points"],
                gt_dense_points=gt_dense
            )
            raw_loss = loss_dict["loss"]
            loss = raw_loss / grad_accum_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()

        if (step + 1) % log_interval == 0 or (step + 1) == num_batches:
            progress_bar.set_postfix(
                total=f"{raw_loss.item():.4f}",
                pose=f"{loss_dict['loss_pose'].item():.4f}",
                sparse_cd=f"{loss_dict['loss_sparse_cd'].item():.4f}",
                dense_cd=f"{loss_dict['loss_dense_cd'].item():.4f}"
            )

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train HORT")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Accumulate gradients across steps for lower VRAM")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_cfg = cfg.get("training", {})
    l_cfg = cfg.get("loss", {})

    dataset_name = args.dataset or cfg.get("dataset", {}).get("name", "obman")
    epochs = args.epochs or t_cfg.get("epochs", 50)
    batch_size = args.batch_size or t_cfg.get("batch_size", 4)
    lr = args.lr or t_cfg.get("lr", 1e-4)
    save_dir = args.save_dir or t_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    print(f"[Train] Using device: {device} | AMP (FP16): {use_amp}")
    print(f"[Train] Dataset: {dataset_name}, Epochs: {epochs}, Batch size: {batch_size}, Grad Accum: {args.grad_accum_steps}, LR: {lr}")

    # Build model
    model = build_model(cfg, device)

    # Trainable parameters (only unfrozen layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Train] Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"Trainable: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    # Optimizer & Scheduler (Adam, LR=1e-4, Cosine decay)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=t_cfg.get("min_lr", 1e-6))

    # Loss
    criterion = HORTLoss(
        lambda_pose=l_cfg.get("lambda_pose", 2.0),
        lambda_sparse_cd=l_cfg.get("lambda_sparse_cd", 2.0),
        lambda_dense_cd=l_cfg.get("lambda_dense_cd", 1.0),
        chunk_size=l_cfg.get("chamfer_chunk_size", 256)
    )

    # Data loaders
    train_dataset = build_dataset(cfg, dataset_name, split="train", max_samples=args.max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if dataset_name == "dummy" else cfg.get("dataset", {}).get("num_workers", 2),
        drop_last=False,
        pin_memory=(device.type == "cuda")
    )

    start_epoch = 1
    best_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"[Train] Resuming checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))

    for epoch in range(start_epoch, epochs + 1):
        loss_epoch = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=t_cfg.get("log_interval", 10)
        )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Checkpoint saving
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": min(best_loss, loss_epoch),
            "config": cfg
        }
        torch.save(latest_ckpt, os.path.join(save_dir, "latest.pt"))

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(latest_ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"[Train] Saved new best model (loss: {best_loss:.4f}) to {save_dir}/best_model.pt")

    print("[Train] Training completed successfully.")


if __name__ == "__main__":
    main()

