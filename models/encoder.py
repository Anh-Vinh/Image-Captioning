import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        trainable: bool = False,
        encoder_dim: int = 512,
    ):
        super().__init__()

        self.backbone_name = backbone

        # 1. Build backbone
        self.backbone, in_features = self._build_backbone(
            backbone, pretrained
        )

        # 2. Freeze if needed
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 3. Projection layer (unifies feature size)
        self.proj = nn.Linear(in_features, encoder_dim)

    def forward(self, x):
        features = self.backbone(x)

        # Handle models that output spatial maps
        if features.ndim == 4:
            features = features.mean(dim=[2, 3])  # GAP

        return self.proj(features)

    def _build_backbone(self, name, pretrained):
        weights = "IMAGENET1K_V1" if pretrained else None

        if name == "resnet18" or name == "resnet50":
            model = models.resnet18(weights=weights if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return model, in_features
