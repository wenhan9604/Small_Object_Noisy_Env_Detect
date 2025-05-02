# Implement KJRD-Net by combining its sub models
import torch
import torch.nn as nn
from models.ffa_net import FFANet
from models.RCAN import RCAN
from models.image_coordination_block import ImageCoordinationBlock
from models.upsample import upsample
from models.masked_autoencoder import MaskedAutoEncoder
from models.diffusion_net import DDPMNet, Diffusion

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
            VIT_weights=None,
            diffusion_weights=None,
            use_diffusion=False,
            device=None
            ):
        super().__init__()
        self.upsample = upsample()
        self.autoencoder = MaskedAutoEncoder(
            chkpt_dir = VIT_weights,
            model_arch = 'mae_vit_large_patch16',
        )  # TODO: check if its called correctly        
        self.ffanet = FFANet(
            num_groups=4,
            num_blocks=2,
            hidden_dim=32,
            kernel_size=3,
            remove_global_skip_connection=False
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

        self.use_diffusion = use_diffusion
        self.device=device

        if self.use_diffusion:
            self.ddpm = DDPMNet(
                hidden_dim=32,
                time_emb_dim=32,
                kernel_size=3
                )
            self.diffusion = Diffusion(timesteps=200)


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
        # if VIT_weights:
        #     self.autoencoder.load_state_dict(torch.load(VIT_weights,weights_only=True))
        #     self.autoencoder.eval()
        #     for param in self.autoencoder.parameters():
        #         param.requires_grad=False
        if self.use_diffusion and diffusion_weights:
            checkpoint = torch.load(diffusion_weights)
            self.ddpm.load_state_dict(checkpoint['model_state_dict'])
            for param in self.ddpm.parameters():
                param.requires_grad=False

    def forward(self, x):
        # Image restoration
        upsample_output = self.upsample(x)
        x_rcan = self.rcan(x)
        if self.use_diffusion:
            x_ffa = self.diffusion.sample(self.ddpm, x_hazy=x, device=self.device)
        else:
            x_ffa = self.ffanet(x)
        x_vit = self.autoencoder(x)
        output = self.image_coordination_block(
            x_rcan, 
            x_ffa, 
            x_vit
            )
        output += upsample_output 

        # Image detection
        detection_output = output
        return detection_output