# Implement KJRD-Net by combining its sub models
import torch
import torch.nn as nn
from models.ffa_net import FFANet
from models.RCAN import RCAN
from models.image_coordination_block import ImageCoordinationBlock
from models.upsample import upsample
# TODO: import autoencoder 


class KJRDNet_wo_detection(nn.Module):
    def __init__(
            self,
            # num_classes,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            ffa_weights=None,
            RCAN_weights=None,
            VIT_weights=None
            ):
        super().__init__()
        self.upsample = upsample
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

        #load pretrained and freeze the weights
        if ffa_weights:
            self.ffanet.load_state_dict(torch.load(ffa_weights,weights_only=True))
            self.ffanet.eval()
            for param in self.ffanet.parameters():
                param.requires_grad=False
        if RCAN_weights:
            self.rcan.load_state_dict(torch.load(RCAN_weights,weights_only=True))
            self.rcan.eval()
            for param in self.rcan.parameters():
                param.requires_grad=False
        if VIT_weights:
            self.autoencoder.load_state_dict(torch.load(VIT_weights,weights_only=True))
            self.autoencoder.eval()
            for param in self.autoencoder.parameters():
                param.requires_grad=False

    def forward(self, x):
        # Image restoration
        upsample = self.upsample(x)
        output = self.image_coordination_block(x)
        output += upsample 
        
        # Image detection
        detection_output = output
        return detection_output