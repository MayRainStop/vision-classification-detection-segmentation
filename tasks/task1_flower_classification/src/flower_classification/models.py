from __future__ import annotations


def require_torchvision():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This command requires torch and torchvision. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return torch, nn, models


RESNET_MODEL_NAMES = {"resnet18", "resnet34", "resnet50", "resnet101"}
ATTENTION_RESNET_MODEL_NAMES = {"resnet18_se", "resnet34_se", "resnet18_cbam", "resnet34_cbam"}
EFFICIENTNET_MODEL_NAMES = {"efficientnet_b0", "efficientnet_b3"}
CONVNEXT_MODEL_NAMES = {"convnext_tiny"}
TRANSFORMER_MODEL_NAMES = {"vit_b_16", "swin_t"}
SUPPORTED_MODEL_NAMES = sorted(
    RESNET_MODEL_NAMES
    | ATTENTION_RESNET_MODEL_NAMES
    | EFFICIENTNET_MODEL_NAMES
    | CONVNEXT_MODEL_NAMES
    | TRANSFORMER_MODEL_NAMES
)


def build_model(model_name: str, num_classes: int = 102, pretrained: bool = True):
    _, nn, models = require_torchvision()
    name = model_name.lower()

    if name in RESNET_MODEL_NAMES | ATTENTION_RESNET_MODEL_NAMES:
        base_name = name.replace("_se", "").replace("_cbam", "")
        constructor = getattr(models, base_name)
        weights = _weights_for_resnet(models, base_name, pretrained)
        model = constructor(weights=weights)
        if name.endswith("_se"):
            model.layer4 = nn.Sequential(model.layer4, make_se_block(model.fc.in_features))
        elif name.endswith("_cbam"):
            model.layer4 = nn.Sequential(model.layer4, make_cbam_block(model.fc.in_features))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown model '{model_name}'. Choose from {', '.join(SUPPORTED_MODEL_NAMES)}."
    )


def make_se_block(channels: int, reduction: int = 16):
    _, nn, _ = require_torchvision()
    hidden = max(channels // reduction, 1)

    class SEBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.excitation(self.pool(x))

    return SEBlock()


def make_cbam_block(channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
    _, nn, _ = require_torchvision()
    hidden = max(channels // reduction, 1)
    padding = spatial_kernel_size // 2

    class CBAMBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.channel_mlp = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            )
            self.channel_gate = nn.Sigmoid()
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=padding, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            channel_attention = self.channel_gate(self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x)))
            x = x * channel_attention
            avg_map = x.mean(dim=1, keepdim=True)
            max_map = x.amax(dim=1, keepdim=True)
            spatial_attention = self.spatial_gate(torch_cat([avg_map, max_map], dim=1))
            return x * spatial_attention

    return CBAMBlock()


def torch_cat(tensors, dim: int):
    torch, _, _ = require_torchvision()
    return torch.cat(tensors, dim=dim)


def _weights_for_resnet(models, name: str, pretrained: bool):
    if not pretrained:
        return None
    if name == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    if name == "resnet34":
        return models.ResNet34_Weights.IMAGENET1K_V1
    if name == "resnet50":
        return models.ResNet50_Weights.IMAGENET1K_V2
    if name == "resnet101":
        return models.ResNet101_Weights.IMAGENET1K_V2
    raise ValueError(f"Unsupported ResNet model '{name}'.")
