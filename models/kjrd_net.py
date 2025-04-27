# Implement KJRD-Net by combining its sub models
import torch
import torch.nn as nn
from models.ffa_net import FFANet
from models.diffusion_net import DDPMNet, Diffusion
from models.RCAN import RCAN
from models.image_coordination_block import ImageCoordinationBlock
from models.upsample import upsample
from models.faster_rcnn import get_custom_faster_rcnn
from models.masked_autoencoder import MaskedAutoEncoder


class KJRDNet(nn.Module):
    def __init__(
            self,
            num_classes,
            upsample,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            ffa_weights=None,
            RCAN_weights=None,
            diffusion_weights=None,
            use_diffusion=False
            ):
        super().__init__()
        self.use_diffusion = use_diffusion
        self.upsample = upsample
        self.autoencoder = MaskedAutoEncoder(
            chkpt_dir = './checkpoint/mae_pretrain_vit_large.pth',
            model_arch = 'mae_vit_large_patch16',
        )  # TODO: check if its called correctly
        
        self.ffanet = FFANet(
            in_channels=3,
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
            vit_token_count=50,
            vit_dim=1024,
            spatial_size=512,
            in_channels=32,
            hidden_dim=32,
            out_channels=3,
            kernel_size=3,
            padding=1
        )
        self.custom_faster_rcnn = get_custom_faster_rcnn(num_classes, icb_channels=out_channels)
        if use_diffusion:
            self.diffusion = DDPMNet(
                hidden_dim=32,
                time_emb_dim=32,
                kernel_size=kernel_size
                )
            self.diffusion = Diffusion(timesteps=200)

        #load pretrained and freeze the weights
        if ffa_weights:
            self.ffanet.load_state_dict(torch.load(ffa_weights))
            for param in self.ffanet.parameters():
                param.requires_grad=False
        if RCAN_weights:
            self.rcan.load_state_dict(torch.load(RCAN_weights))
            for param in self.rcan.parameters():
                param.requires_grad=False
        if use_diffusion and diffusion_weights:
            checkpoint = torch.load(diffusion_weights)
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])
            for param in self.diffusion.parameters():
                param.requires_grad=False

    def forward(self, x):
        # Image restoration
        upsample = self.upsample(x)
        x_rcan = self.rcan(x)
        if self.use_diffusion:
            x_ffa = self.diffusion.sample(self.diffusion, x_hazy=x, device=self.device)
        else:
            x_ffa = self.ffanet(x)
        x_vit = self.autoencoder(x)
        output = self.image_coordination_block(
            x_rcan, 
            x_ffa, 
            x_vit
            )
        output += upsample 
        
        # Image detection
        detection_output = self.custom_faster_rcnn(output)
        return detection_output