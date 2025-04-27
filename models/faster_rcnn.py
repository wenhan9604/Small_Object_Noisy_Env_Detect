import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class faster_rcnn(nn.Module):
    
    def __init__(self, num_classes=17, freeze_backbone=True):
        super().__init__()
        # Standard Faster R-CNN
        self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        #replace detection head and for classification task
        # Get input features from existing classifier
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        # Replace default COCO predictor
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, images, targets=None):        
        return self.detector(images, targets)
