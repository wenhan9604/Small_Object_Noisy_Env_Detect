from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


from torch.utils.data import Dataset
from PIL import Image
import os


class DotahFfanetDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        hazy_path = os.path.join(self.hazy_dir, filename)
        clear_path = os.path.join(self.clear_dir, filename)

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)

        return hazy_img, clear_img
