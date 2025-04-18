from torch.utils.data import Dataset
from PIL import Image
import os

class DotahRCANDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform_hazy=None,transform_clear=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform_hazy = transform_hazy
        self.transform_clear = transform_clear
        self.filenames = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        hazy_path = os.path.join(self.hazy_dir, filename)
        clear_path = os.path.join(self.clear_dir, filename)

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        if self.transform_hazy:
            hazy_img = self.transform_hazy(hazy_img)
        if self.transform_clear:
            clear_img = self.transform_clear(clear_img)
            

        # print(hazy_img.shape)
        # print(clear_img.shape)

        return hazy_img, clear_img
