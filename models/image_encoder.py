import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """DINOv2-Large Image Encoder.
    - Input: 224x224 RGB image (B, 3, 224, 224)
    - Output: 257x1024 tokens (B, 257, 1024) [CLS token + 256 patch tokens]
    - Fine-tuning: Freeze early DINOv2 layers (blocks 0..11 and patch embeddings),
      fine-tune final 12 transformer layers (blocks 12..23) and final norm.
    """
    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_dinov2",
        pretrained: bool = True,
        img_size: int = 224,
        embed_dim: int = 1024,
        num_frozen_layers: int = 12
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_frozen_layers = num_frozen_layers

        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0  # feature extraction only
            )
        except Exception as e:
            print(f"[ImageEncoder] Warning: Failed to load pretrained weights ({e}). Initializing randomly.")
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                img_size=img_size,
                num_classes=0
            )

        if hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze patch embedding, pos embedding, and the first `num_frozen_layers` transformer blocks."""
        # Freeze patch embedding
        if hasattr(self.backbone, "patch_embed"):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False

        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            self.backbone.pos_embed.requires_grad = False

        if hasattr(self.backbone, "cls_token") and self.backbone.cls_token is not None:
            self.backbone.cls_token.requires_grad = False

        # Freeze first num_frozen_layers blocks
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            freeze_until = min(self.num_frozen_layers, num_blocks)
            for i in range(freeze_until):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False

            # Ensure subsequent blocks have gradients enabled
            for i in range(freeze_until, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Norm layer trainable
        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) RGB image normalized with ImageNet stats.
        Returns:
            fv: (B, 257, 1024) tokens [CLS token at index 0, 256 patch tokens at indices 1..256].
        """
        assert x.dim() == 4 and x.shape[1] == 3 and x.shape[2] == self.img_size and x.shape[3] == self.img_size, \
            f"Expected input shape (B, 3, {self.img_size}, {self.img_size}), got {x.shape}"

        feat = self.backbone.forward_features(x)  # (B, 257, 1024)
        assert feat.shape[1] == 257 and feat.shape[2] == self.embed_dim, \
            f"Expected feature shape (B, 257, {self.embed_dim}), got {feat.shape}"
        return feat

