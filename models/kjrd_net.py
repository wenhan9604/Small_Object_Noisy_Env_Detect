# Implement KJRD-Net by combining its sub models
import torch
import torch.nn as nn
from models.ffa_net import FFANet
from models.image_coordination_block import ImageCoordinationBlock
from models.faster_rcnn import get_custom_faster_rcnn
# TODO: import upsampling
# TODO: import autoencoder 
# TODO: import models


class KJRDNet(nn.Module):
    def __init__(
            self,
            num_classes,
            upsample,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
            ):
        super().__init__()
        self.upsample = upsample  # TODO: check if its called correctly
        self.autoencoder = autoencoder()  # TODO: check if its called correctly
        self.ffanet = FFANet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
            )
        self.rcan = rcan()  # TODO: check if its called correctly
        self.image_coordination_block = ImageCoordinationBlock(
            autoencoder=self.autoencoder,
            rcan=self.rcan,
            ffanet=self.ffanet,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.custom_faster_rcnn = get_custom_faster_rcnn(num_classes, icb_channels=out_channels)

    def forward(self, x):
        # Image restoration
        upsample = self.upsample(x)
        output = self.image_coordination_block(x)
        output += upsample  # TODO: check if shapes match
        
        # Image detection
        detection_output = self.custom_faster_rcnn(output)
        return detection_output