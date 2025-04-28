import torch
import torch.nn as nn
import numpy as np
from mae import models_mae

class MaskedAutoEncoder(nn.Module):
    def __init__(
            self,
            chkpt_dir = './checkpoint/mae_pretrain_vit_large.pth',
            model_arch = 'mae_vit_large_patch16',
            imagenet_mean = np.array([0.485, 0.456, 0.406]),
            imagenet_std = np.array([0.229, 0.224, 0.225])
            ):

        super().__init__()
        self.model = self.prepare_model(chkpt_dir, model_arch)
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=False


    def forward(self, img):

        processed_img = self.ViT_image_preprocessing(img)
        embedding = self.image_encoding(processed_img, self.model)

        return embedding
    
    def prepare_model(self, chkpt_dir, arch='mae_vit_base_patch16'):

        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model
    
    def ViT_image_preprocessing(self, img):

        img = img.resize((224, 224))
        img = np.array(img) / 255.

        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - self.imagenet_mean
        img = img / self.imagenet_std

    def image_encoding(self, img, model):

        x = torch.tensor(img)

        #make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask, latent = model(x.float(), mask_ratio=0.75)
        # y = model.unpatchify(y)
        # y = torch.einsum('nchw->nhwc', y).detach().cpu()

        return latent


