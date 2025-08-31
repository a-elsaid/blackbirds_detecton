from torchvision.models.detection import fasterrcnn_resnet50_fpn,  retinanet_resnet50_fpn 
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as frcnn_weights
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights as retinanet_weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import ResNet50_Weights as resnet_weights 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import ops
import torch.nn as nn
from typing import Dict

MODELS = {
        "frcnn-resnet": fasterrcnn_resnet50_fpn,
        "retinanet": retinanet_resnet50_fpn
         }

MODEL_WEIGHT = {
                "frcnn-resnet": frcnn_weights.DEFAULT,
                "retinanet": retinanet_weights.DEFAULT,
                }

class CustomModel(nn.Module):
    """
    Wraps Faster R-CNN, applies your 'customed_model' edits, and exposes extras.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 4,
        image_mean=(0.485, 0.456, 0.406, 0.406),
        image_std=(0.229, 0.224, 0.225, 0.225),
        max_size: int = 3000,
        use_backbone_weights: bool = True,
        use_model_weights: bool = True,
        type: str = "frcnn-resnet",
        classes: Dict[str, int] = None,
    ):
        super().__init__()

        core = MODELS[type](
            weights=MODEL_WEIGHT[type] if use_model_weights else None,
            weights_backbone=resnet_weights.DEFAULT if use_backbone_weights else None,
            progress=True,
        )

        # --- Apply transform customization ---
        t = core.transform
        core.transform = GeneralizedRCNNTransform(
            min_size=t.min_size,
            max_size=max_size,
            image_mean=list(image_mean),
            image_std=list(image_std),
        )

        # --- Apply your backbone (ResNet-50 body assumed) ---
        body = core.backbone.body
        # conv1 to accept custom channels
        body.conv1 = nn.Conv2d(in_channels, 256, kernel_size=7, stride=2, padding=3, bias=False)
        body.bn1   = ops.FrozenBatchNorm2d(256, eps=0.0)

        blk = body.layer1[0]
        blk.conv1 = nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3, bias=False)
        blk.bn1   = ops.FrozenBatchNorm2d(256, eps=0.0)
        blk.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        blk.bn2   = ops.FrozenBatchNorm2d(128, eps=0.0)
        blk.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        blk.bn3   = ops.FrozenBatchNorm2d(256, eps=0.0)
        blk.downsample[0] = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)

        # --- Replace the classifier head (num_classes + background) ---
        in_feat = core.roi_heads.box_predictor.cls_score.in_features
        core.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

        # Store the core model and extra attributes
        self.core = core
        self.in_channels = in_channels

        # Store the training-class mapping -- Used for prediction
        self.classes = classes if classes is not None else {}

    # --- Forward simply delegates to the core detector ---
    def forward(self, *args, **kwargs):
        return self.core(*args, **kwargs)

    # --- Convenience properties / methods you asked for ---
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_grad_params(self) -> int:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def input_channels(self) -> int:
        return self.in_channels

    def get_transform_stats(self):
        t = self.core.transform
        return {
            "min_size": t.min_size,
            "max_size": t.max_size,
            "image_mean": t.image_mean,
            "image_std": t.image_std,
        }

    def set_max_size(self, max_size: int):
        t = self.core.transform
        self.core.transform = GeneralizedRCNNTransform(
            min_size=t.min_size, max_size=max_size,
            image_mean=t.image_mean, image_std=t.image_std
        )
        self.max_size = max_size

    def freeze_backbone(self, freeze: bool = True):
        for p in self.core.backbone.parameters():
            p.requires_grad = not freeze
        return self

    def replace_head(self, num_classes: int):
        """Replace the detector head after init (e.g., new dataset)."""
        in_feat = self.core.roi_heads.box_predictor.cls_score.in_features
        self.core.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

    # Optionally delegate attribute access so you can do self.backbone, etc.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.core, name)
