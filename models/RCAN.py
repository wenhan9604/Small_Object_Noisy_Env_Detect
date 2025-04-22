import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint ,checkpoint_sequential

class RCAN(nn.Module):
    # def __init__(self, num_of_image_channels=3, num_of_RG=10, num_of_RCAB=20, num_of_features=64, upscale_factor=2): ##original param according to paper
    def __init__(self, num_of_image_channels=3, num_of_RG=5, num_of_RCAB=10, num_of_features=32, upscale_factor=2):
        super().__init__()

        #shallow feature extraction (conv layer)
        self.shallow_feature=nn.Conv2d(in_channels=num_of_image_channels,out_channels=num_of_features,kernel_size=3,padding=1)

        #RIR block
        self.RIR=RIR(num_of_RCAB=num_of_RCAB,num_of_features=num_of_features,num_of_RG=num_of_RG)

        #upsampling module, to implement using ESPCNN as per What Li et al used.
        #For ESPCNN, refer to paper Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (https://arxiv.org/abs/1609.05158)
        #Dedicate torch module for this, read nn.PixelShuffle
        self.upscale=nn.Sequential(
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features*(upscale_factor**2),kernel_size=3,padding=1),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

        #reconstruction layer (conv layer)
        self.reconstruct=nn.Conv2d(in_channels=num_of_features,out_channels=num_of_image_channels,kernel_size=3,padding=1)

    def forward(self, x):
        x=checkpoint(self.shallow_feature,x,use_reentrant=False)
        x=checkpoint(self.RIR,x,use_reentrant=False)
        x=checkpoint_sequential(self.upscale,segments=2,input=x,use_reentrant=False)
        x=checkpoint(self.reconstruct,x,use_reentrant=False)

        # x=self.shallow_feature(x)
        # x=self.RIR(x)
        # x=self.upscale(x)
        # x=self.reconstruct(x)
        return x

class RIR(nn.Module):
    def __init__(self,num_of_RCAB, num_of_features, num_of_RG):
        super().__init__()
        
        self.layers=nn.Sequential(
            *[RG(num_of_RCAB=num_of_RCAB,num_of_features=num_of_features) for _ in range(num_of_RG)],
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features,kernel_size=3,padding=1)
        )
        
    def forward(self,x):
        #takes care of long skip connection
        return x+self.layers(x)

class RG(nn.Module):
    def __init__(self, num_of_RCAB, num_of_features):
        super().__init__()
        
        self.layers=nn.Sequential(
            *[RCAB(num_of_features=num_of_features) for _ in range(num_of_RCAB)],
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features,kernel_size=3,padding=1)
        )        
    def forward(self,x):
        #takes care of short skip connection
        return x+self.layers(x)

class RCAB(nn.Module):
    def __init__(self,num_of_features,reduction=16):
        super().__init__()
        
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features,kernel_size=3,padding=1),
            ChannelAttention(num_of_features=num_of_features,reduction=reduction)
        )
    def forward(self,x):
        #takes care of skip connection
        return x+self.layers(x)

class ChannelAttention(nn.Module):
    def __init__(self,num_of_features,reduction):
        super().__init__()
        self.layers=nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #Paper says pool to 1x1xC
            nn.Conv2d(in_channels=num_of_features,out_channels=num_of_features//reduction,kernel_size=1,padding=0), #padding is 0 because already reduced to 1x1xC
            nn.ReLU(),
            nn.Conv2d(in_channels=num_of_features//reduction,out_channels=num_of_features,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self,x):
        #takes care of skip connection
        return torch.mul(x,self.layers(x))