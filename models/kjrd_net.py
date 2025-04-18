# Implement KJRD-Net by combining its sub models
import torch
import torch.nn as nn
from models.ffa_net import FFANet
from models.RCAN import RCAN
from models.image_coordination_block import ImageCoordinationBlock
from models.upsample import upsample
from models.faster_rcnn import get_custom_faster_rcnn
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
        self.rcan = RCAN(
            num_of_image_channels=3,
            num_of_RG=5,
            num_of_RCAB=10,
            num_of_features=32,
            upscale_factor=2
            )
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
        output += upsample 
        
        # Image detection
        detection_output = self.custom_faster_rcnn(output)
        return detection_output