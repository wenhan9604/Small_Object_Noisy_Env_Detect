import os, torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data_utils import get_device, set_seed
from datasets.dotah_ffa_net_dataset import DotahFfanetDataset

class Trainer:
    def __init__(self, config, output_dir=None, device=None):
        self.config = config
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.num_epochs = self.config.train.n_epochs
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
        if self.dataset.lower() == 'dotah_ffa_net':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

            def make_split(split):
                hazy_path = f'./datasets/dotah/{split}/hazy'
                clear_path = f'./datasets/dotah/{split}/clear'
                if not (os.path.isdir(hazy_path) and os.path.isdir(clear_path)):
                    raise FileNotFoundError(f"Missing hazy or clear directory for: {split}")
                return DotahFfanetDataset(hazy_dir=hazy_path, clear_dir=clear_path, transform=transform)

            self.trainset = make_split('train')
            self.validationset = make_split('val')
            self.testset = make_split('test')

        else:
            raise NotImplementedError("Dataset is not supported")

        self.train_loader = DataLoader(self.trainset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=self.num_workers)
        
        self.val_loader = DataLoader(self.validationset, 
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
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_class, model_path, map_location='cpu'):
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        return model

    def train(self):
        raise NotImplementedError("train function is not implemented")
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Epoch [{epoch+1}] Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def test(self, epoch):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.val_loader)
        print(f"Epoch [{epoch+1}] Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss


        
