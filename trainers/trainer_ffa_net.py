import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.ffa_net import FFANet
from trainers.trainer_base import Trainer

class FFANetTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        self.model = FFANet()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.L1Loss()

    def train(self):
        self.model.train()
        running_loss = 0.0
        start = time.time()
        for epoch in range(self.num_epochs):
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")
            running_loss = 0.0

        print(f"Completed in {(time.time()-start):.3f}.")
        self.save_model(self.model, f'{self.output_dir}/ffa_net_{self.dataset.lower()}.pth')
        