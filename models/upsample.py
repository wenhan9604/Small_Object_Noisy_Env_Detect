import torch.nn as nn

#implement upsample using https://arxiv.org/abs/1609.05158
#Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
class upsample(nn.Module):
    def __init__(self,upscale_factor=2):
        super().__init__()

        self.project = nn.Conv2d(3, 4, kernel_size=1)
        self.upscale=nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        x = self.project(x)
        
        return self.upscale(x)