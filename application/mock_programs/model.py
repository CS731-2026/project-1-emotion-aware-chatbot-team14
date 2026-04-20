"""
CS731 — Emotion Recognition Model Definitions
===============================================
Builds all architectures compared in the project using the timm library.

Models benchmarked:
  Group 15 (Frienderly):   efficientnet_b0, efficientnet_b3,
                            swin_tiny_patch4_window7_224, mobilevit_s, convnext_tiny
  Team 7   (ChatBox):      efficientnet_b0, efficientnetv2_s (+ custom head), mobilenet_v3_small

The ChatBox_V1 variant replaces the standard classifier head with a deeper
multi-layer MLP (BatchNorm → ReLU → Dropout) for better feature separation.

Usage
-----
  from models.model import build_model, AVAILABLE_MODELS
  model = build_model('swin_tiny', num_classes=6)
  model = build_model('chatbox_v1', num_classes=7)
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print('[WARN] timm not installed. Run: pip install timm')


# ── Model registry ────────────────────────────────────────────────────────────

# Maps friendly short names → exact timm model strings
TIMM_MODEL_MAP = {
    # Group 15 models
    'efficientnet_b0':  'efficientnet_b0',
    'efficientnet_b3':  'efficientnet_b3',
    'swin_tiny':        'swin_tiny_patch4_window7_224',
    'mobilevit_s':      'mobilevit_s',
    'convnext_tiny':    'convnext_tiny',
    # Team 7 models
    'efficientnetv2_s': 'efficientnetv2_s',
    'mobilenet_v3':     'mobilenetv3_small_100',
    # Additional options
    'resnet50':         'resnet50',
    'vit_tiny':         'vit_tiny_patch16_224',
}

AVAILABLE_MODELS = list(TIMM_MODEL_MAP.keys()) + ['chatbox_v1']


# ── Standard timm model ───────────────────────────────────────────────────────

def build_standard_model(model_name: str, num_classes: int,
                          pretrained: bool = True) -> nn.Module:
    """
    Build a timm model with its default classification head replaced
    to match num_classes.
    """
    if not TIMM_AVAILABLE:
        raise ImportError('timm is required. pip install timm')

    timm_name = TIMM_MODEL_MAP.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=pretrained,
                               num_classes=num_classes)
    return model


# ── ChatBox_V1 — custom multi-layer classifier ────────────────────────────────

class ChatBoxV1(nn.Module):
    """
    EfficientNet V2-S backbone + custom 3-layer MLP classifier head.
    Matches the Team 7 architecture that achieved 81% val accuracy.

    Architecture (classifier):
        Linear(1280 → 1024) → BatchNorm → ReLU → Dropout(0.3)
        Linear(1024 → 512)  → BatchNorm → ReLU → Dropout(0.2)
        Linear(512  → num_classes)

    The deeper head captures more complex feature interactions than a
    single linear layer, improving discrimination between similar emotions
    (e.g. fear vs surprise, anger vs disgust).
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError('timm is required. pip install timm')

        # Load EfficientNet V2-S backbone
        backbone = timm.create_model('efficientnetv2_s', pretrained=pretrained,
                                      num_classes=0)   # num_classes=0 → identity head
        in_features = backbone.num_features             # 1280 for efficientnetv2_s

        self.backbone   = backbone
        self.classifier = nn.Sequential(
            # Layer 1: expansion
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # Layer 2: intermediate
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # Layer 3: output
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # (B, 1280)
        return self.classifier(features) # (B, num_classes)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Main factory ──────────────────────────────────────────────────────────────

def build_model(model_name: str,
                num_classes: int = 6,
                pretrained:  bool = True) -> nn.Module:
    """
    Build and return a model ready for training.

    Args:
        model_name:  short name from AVAILABLE_MODELS, or exact timm string
        num_classes: number of emotion classes (6 for Ekman-6, 7 for +neutral)
        pretrained:  load ImageNet weights for the backbone

    Returns:
        nn.Module
    """
    name = model_name.lower().replace('-', '_')

    if name == 'chatbox_v1':
        model = ChatBoxV1(num_classes=num_classes, pretrained=pretrained)
    else:
        model = build_standard_model(name, num_classes, pretrained)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Built {model_name}: {n_params:,} trainable parameters '
          f'({num_classes} classes)')
    return model


# ── Model summary helper ──────────────────────────────────────────────────────

def model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224),
                  device: str = 'cpu') -> None:
    """Print a quick parameter and output-shape summary."""
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        out = model(x)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Input  : {tuple(x.shape)}')
    print(f'  Output : {tuple(out.shape)}')
    print(f'  Params : {total:,} total | {trainable:,} trainable')


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Available models:', AVAILABLE_MODELS)
    print()
    for name in ('swin_tiny', 'convnext_tiny', 'efficientnet_b0', 'chatbox_v1'):
        print(f'── {name} ──')
        try:
            m = build_model(name, num_classes=6, pretrained=False)
            model_summary(m)
        except Exception as e:
            print(f'  [ERROR] {e}')
        print()
