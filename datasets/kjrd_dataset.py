from torch.utils.data import Dataset
from PIL import Image
import os, torch

class KJRDDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, labels_dir, transform_clear=None, disable_difficult=False, detector_input_image_size=512):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.labels_dir=labels_dir
        self.transform_clear=transform_clear
        
        self.filenames = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])

        self.detector_in_image_size=detector_input_image_size
        self.disable_difficult = disable_difficult

        self.class_map={
            'plane':1,
            'ship':2,
            'storage-tank':3,
            'baseball-diamond':4,
            'tennis-court':5,
            'basketball-court':6,
            'ground-track-field':7,
            'harbor':8,
            'bridge':9,
            'large-vehicle':10,
            'small-vehicle':11,
            'helicopter':12,
            'roundabout':13,
            'soccer-ball-field':14,
            'swimming-pool':15,
            'container-crane':16
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        hazy_path = os.path.join(self.hazy_dir, filename)
        clear_path = os.path.join(self.clear_dir, filename)
        label_path = os.path.join(self.labels_dir, filename)

        hazy_img = Image.open(hazy_path).convert('RGB') #this will be 256x256
        clear_img = Image.open(clear_path).convert('RGB') #this will be full size, variable

        clear_width,clear_height=clear_img.size
        x_ratio, y_ratio = self.detector_in_image_size/clear_width, self.detector_in_image_size/clear_height

        if self.transform_clear:
            clear_img = self.transform_clear(clear_img)
        
        boxes, labels=[],[]
        with open(label_path, 'r') as f:
            for line in f:
                difficulty = int(line.split()[-1])
                if self.disable_difficult and difficulty == 1:
                    continue #skip if this is a difficult row
                boxes.append(self.rescale_labels(self.OBB_to_AABB(line),x_ratio,y_ratio))
                class_name=line.split()[-2]
                labels.append(self.class_map[class_name.lower()])
            
        boxes = torch.as_tenor(boxes, dtype=torch.int64)
        labels = torch.as_tenor(labels, dtype=torch.int64)

        return hazy_img, {'ref_image':clear_img, 'object_labels':{'boxes':boxes,'labels':labels}}

    def rescale_labels(self,c,x_ratio,y_ratio):
        return [round(c[0]/x_ratio), round(c[1]/y_ratio), round(c[2]/x_ratio), round(c[3]/y_ratio)]

    def OBB_to_AABB(self,line):
        coords = list(map(float,line.split()[:8]))
        x_coords=coords[::2]
        y_coords=coords[1::2]
        return [min(x_coords), min(y_coords),max(x_coords),max(y_coords)]