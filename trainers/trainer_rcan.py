import time
# import torch
import torch.nn as nn
# import torch.optim as optim
from models.RCAN import RCAN
from trainers.trainer_base import Trainer

class RCANTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        
        self.model = RCAN().to(self.device)
        self.optimizer = self._init_optimizer(net=self.model)
        self.criterion = nn.L1Loss()

    def train(self):
        print(f"Training RCAN")

        
        start = time.time()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            #sets model in training mode. does not perform the train
            self.model.train()

            for i, (inputs, targets) in enumerate(self.train_loader):
                print('Training batch:', i)
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
            
            avg_train_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_loss = self.validate(epoch)

        self.save_model(self.model, f'{self.output_dir}/rcan_{self.dataset.lower()}.pth')
        print(f"Completed in {(time.time()-start):.3f}.")

    

        