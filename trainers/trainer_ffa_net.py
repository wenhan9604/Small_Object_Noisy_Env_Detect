import os
import time
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from models.ffa_net import FFANet
from trainers.trainer_base import Trainer

class FFANetTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        self.hidden_dim = self.config.train.hidden_dim
        self.num_groups = self.config.train.num_groups
        self.num_blocks = self.config.train.num_attention_blocks
        self.kernel_size = self.config.train.kernel_size
        self.remove_global_skip_connection = self.config.train.remove_global_skip_connection
        self.model = FFANet(
            num_groups=self.num_groups,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            remove_global_skip_connection=self.remove_global_skip_connection
            ).to(self.device)
        self.optimizer = self._init_optimizer(net=self.model)
        self.criterion = nn.L1Loss()

    def train(self):
        print(f"Training FFA-Net")
        self.model.train()
        start = time.time()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()

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
            
            avg_train_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_loss = self.validate(epoch)

        self.save_model(self.model, f'{self.output_dir}/ffa_net_{self.dataset.lower()}.pth')

        # Save output samples
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            print("Saving output samples...")
            self.save_output_images(self.model, self.test_loader, device=self.device, save_dir=f"{self.output_dir}", num_samples=2)

        print(f"Completed in {(time.time()-start):.3f}.")

    def save_output_images(self, model, dataloader, device, save_dir="output_images", num_samples=2):
        
        def label_image(image, label, font_size=20):
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            draw.text((10, 10), label, font=font, fill=(255, 0, 0))
            return image

        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        to_pil = T.ToPILImage()

        with torch.no_grad():
            count = 0
            for hazy, clear in dataloader:
                hazy = hazy.to(device)
                output = model(hazy)
                output = torch.clamp(output, 0, 1)

                for i in range(num_samples):
                    if count >= num_samples:
                        break
                    print(f'Saving image {i}')
                    hazy_img = label_image(to_pil(hazy[i].cpu()), "Hazy")
                    dehazed_img = label_image(to_pil(output[i].cpu()), "Dehazed")
                    gt_img = label_image(to_pil(clear[i].cpu()), "Ground Truth")

                    width, height = hazy_img.size
                    combined = Image.new("RGB", (width * 3, height))
                    combined.paste(hazy_img, (0, 0))
                    combined.paste(dehazed_img, (width, 0))
                    combined.paste(gt_img, (width * 2, 0))

                    combined.save(os.path.join(save_dir, f"output_comparison_{i}.png"))
                    count += 1

        print(f"Saved {i} output samples to '{save_dir}'")



    

        