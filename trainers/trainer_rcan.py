import time, torch, os
import torch.nn as nn
# import torch.optim as optim
from models.RCAN import RCAN
from trainers.trainer_base import Trainer
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

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

        # Save output samples
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            print("Saving output samples...")
            self.save_output_images(self.model, self.test_loader, device=self.device, save_dir=f"{self.output_dir}", num_samples=2)

        print(f"Completed in {(time.time()-start):.3f}.")
    
    def generate_sample(self,model_weights_path):
        model=self.load_model2(self.model,model_weights_path)

        # Save output samples
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            print("Saving output samples...")
            self.save_output_images(model, self.test_loader, device=self.device, save_dir=f"{self.output_dir}", num_samples=2)

    def save_output_images(self, model, dataloader, device, save_dir="output_images", num_samples=2):
        
        def label_image(image, label, font_size=20):
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arialbd.ttf", 20)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text((0, 0), label, font=font, fill='white')
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
                    rcaned_img = label_image(to_pil(output[i].cpu()).resize((256,256),Image.LANCZOS), "RCAN output")
                    gt_img = label_image(to_pil(clear[i].cpu()).resize((256,256),Image.LANCZOS), "Ground Truth")

                    width, height = hazy_img.size
                    combined = Image.new("RGB", (width * 3, height))
                    combined.paste(hazy_img, (0, 0))
                    combined.paste(rcaned_img, (width, 0))
                    combined.paste(gt_img, (width * 2, 0))

                    combined.save(os.path.join(save_dir, f"output_comparison_rcan_{i}.png"))
                    count += 1

        print(f"Saved {i} output samples to '{save_dir}'")

    

        