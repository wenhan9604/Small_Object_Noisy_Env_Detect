import os, torch, shutil, random, numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from utils.data_utils import get_device, set_seed



class Trainer:
    def __init__(self, config, output_dir=None, device=None):
        self.config = config
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.dataset = self.config.data.dataset

        set_seed(seed=42)

        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)

        if output_dir is None:
            self.output_dir = f"./outputs/{self.config.network.model.lower()}"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True,)

        # Initialize datasets
        if self.dataset.lower() == 'xxx':  # TODO: replace with actual dataset
            self.trainset = None
            self.testset = None
        else:
           print('unsupported dataset')

        self.train_loader = DataLoader(self.trainset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.testset, 
                                      batch_size=self.batch_size, 
                                      shuffle=False, 
                                      num_workers=self.num_workers)

    def _init_optimizer(self, net):
        if self.config.optimizer.type.lower() == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr = self.lr, betas=(0.9,0.999), weight_decay=self.config.optimizer.weight_decay)
        else:
            raise ValueError("unsupported optimizer. use sgd or adamw")
        return optimizer

    @staticmethod
    def save_model(model, model_path):
        model_s = torch.jit.script(model)
        model_s.save(model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path, map_location='cpu'):
        model = torch.jit.load(model_path, map_location=map_location)
        return model


