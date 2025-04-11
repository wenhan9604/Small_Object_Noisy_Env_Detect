import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from image_coordination_block import ImageCoordinationBlock


# Create custom class to add ICB feature extraction to FasterRCNN as mentioned in the paper
class CustomFusionBackbone(nn.Module):
    def __init__(self, image_coordination_block, icb_channels=64):
        super().__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.image_coordination_block = image_coordination_block
        # Transform ICB output to match fpn
        self.feature_transform = nn.Sequential(
            nn.Conv2d(icb_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.image_coordination_block(x)
        transformed_features = self.feature_transform(features)
        resnet_features = self.backbone.body(x)
        resnet_features['0'] = resnet_features['0'] + transformed_features
        return resnet_features
    

def get_custom_faster_rcnn(num_classes, icb_channels=64):
    image_coordination_block = ImageCoordinationBlock()
    backbone = CustomFusionBackbone(
        image_coordination_block=image_coordination_block,
        icb_channels=icb_channels
        )
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model