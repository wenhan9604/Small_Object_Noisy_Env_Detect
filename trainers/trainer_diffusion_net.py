import os
import time
from PIL import Image, ImageDraw, ImageFont
import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from models.diffusion_net import DDPMNet, Diffusion
from trainers.trainer_base import Trainer
 
class DDPMNetTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        self.num_epochs = self.config.train.n_epochs
        self.hidden_dim = self.config.train.hidden_dim
        self.time_emb_dim = self.config.train.time_emb_dim
        self.kernel_size = self.config.train.kernel_size
        self.timesteps = self.config.train.timesteps
        self.beta = self.config.train.beta
        self.image_loss_weight = self.config.train.image_loss_weight
        self.color_loss_weight = self.config.train.color_loss_weight
        self.model = DDPMNet(
            hidden_dim=self.hidden_dim,
            time_emb_dim=self.time_emb_dim,
            kernel_size=self.kernel_size
            ).to(self.device)
        self.diffusion = Diffusion(timesteps=self.timesteps)
        self.optimizer = self._init_optimizer(net=self.model)
        resume_path = getattr(self.config.train, "resume_from", None)
        if resume_path and os.path.isfile(resume_path):
            ckpt = torch.load(self.config.train.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from {self.config.train.resume_from}, starting at epoch {self.start_epoch}")
        else:
            self.start_epoch = 0

    def train(self):
        start = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            running_loss = 0.0
            running_v_loss = 0.0
            running_image_loss = 0.0
            running_color_loss = 0.0

            self.model.train()

            for i, (x_hazy, x_clean) in enumerate(self.train_loader):
                x_hazy, x_clean = x_hazy.to(self.device), x_clean.to(self.device)
                self.optimizer.zero_grad()

                t = torch.randint(0, self.diffusion.timesteps, (x_clean.size(0),), device=self.device)
                x_t, noise = self.diffusion.add_noise(x_clean, t)

                v_pred = self.model(x_t, x_hazy, t)

                alpha_hat = self.diffusion.alpha_hats.to(x_t.device)
                alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)
                sqrt_ah = torch.sqrt(alpha_hat_t)
                sqrt_oma = torch.sqrt(1.0 - alpha_hat_t)

                x0_pred = sqrt_ah * x_t - sqrt_oma * v_pred
                x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

                # Compute loss components
                loss_v = F.mse_loss(v_pred, sqrt_ah * noise - sqrt_oma * x_clean)
                loss_image = F.l1_loss(x0_pred, x_clean)
                lab_pred, lab_gt = K.color.rgb_to_lab(x0_pred), K.color.rgb_to_lab(x_clean)
                lab_pred[:, 0] /= 100.0
                lab_gt[:, 0] /= 100.0
                lab_pred[:, 1:] /= 128.0
                lab_gt[:, 1:] /= 128.0
                loss_color = F.l1_loss(lab_pred, lab_gt)

                loss = loss_v + self.image_loss_weight * loss_image + self.color_loss_weight * loss_color

                # Backprop + optimize
                loss.backward()
                self.optimizer.step()

                # Track losses
                running_loss += loss.item()
                running_v_loss += loss_v.item()
                running_image_loss += loss_image.item()
                running_color_loss += loss_color.item()

            avg_train_loss = running_loss / len(self.train_loader)
            avg_v_loss = running_v_loss / len(self.train_loader)
            avg_image_loss = running_image_loss / len(self.train_loader)
            avg_color_loss = running_color_loss / len(self.train_loader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] | "
                  f"Total Loss: {avg_train_loss:.4f} | "
                  f"V Loss: {avg_v_loss:.4f} | "
                  f"Image Loss: {avg_image_loss:.4f} | "
                  f"Color Loss: {avg_color_loss:.4f}")

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                self.validate(epoch)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch + 1
        }, f"{self.output_dir}/diffusion_net_{self.dataset.lower()}.pth")
        print(f"Saved model to {self.output_dir}/diffusion_net_{self.dataset.lower()}.pth")

        # Save output samples
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            print("Saving output samples...")
            self.save_output_images(self.model, self.test_loader, device=self.device,
                                    save_dir=f"{self.output_dir}", num_samples=2)

        print(f"Completed in {(time.time() - start):.3f}.")

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_hazy, x_clean in self.val_loader:
                x_hazy, x_clean = x_hazy.to(self.device), x_clean.to(self.device)

                t = torch.randint(0, self.diffusion.timesteps, (x_clean.size(0),), device=self.device)
                x_t, noise = self.diffusion.add_noise(x_clean, t)
                v_pred = self.model(x_t, x_hazy, t)

                alpha_hat = self.diffusion.alpha_hats.to(x_t.device)
                alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)

                sqrt_ah = torch.sqrt(alpha_hat_t)
                sqrt_oma = torch.sqrt(1.0 - alpha_hat_t)

                x0_pred = sqrt_ah * x_t - sqrt_oma * v_pred
                x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

                # Hybrid loss
                loss_v = F.mse_loss(v_pred, sqrt_ah * noise - sqrt_oma * x_clean)
                loss_image = F.l1_loss(x0_pred, x_clean)
                lab_pred, lab_gt = K.color.rgb_to_lab(x0_pred), K.color.rgb_to_lab(x_clean)
                lab_pred[:, 0] /= 100.0
                lab_gt[:, 0] /= 100.0
                lab_pred[:, 1:] /= 128.0
                lab_gt[:, 1:] /= 128.0
                loss_color  = F.l1_loss(lab_pred, lab_gt)
                loss = loss_v + self.image_loss_weight * loss_image + self.color_loss_weight * loss_color
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Epoch [{epoch+1}] Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

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
                output = self.diffusion.sample(model, x_hazy=hazy, device=device)
                output = torch.clamp(output, 0, 1)

                for i in range(num_samples):
                    if count >= num_samples:
                        break
                    hazy_img = label_image(to_pil(hazy[i].cpu()), "Hazy")
                    dehazed_img = label_image(to_pil(output[i].cpu()), "Dehazed")
                    gt_img = label_image(to_pil(clear[i].cpu()), "Ground Truth")

                    width, height = hazy_img.size
                    combined = Image.new("RGB", (width * 3, height))
                    combined.paste(hazy_img, (0, 0))
                    combined.paste(dehazed_img, (width, 0))
                    combined.paste(gt_img, (width * 2, 0))
                    print(f'Saving image {i}')
                    combined.save(os.path.join(save_dir, f"output_comparison_{i}.png"))
                    count += 1

        print(f"Saved {num_samples} output samples to '{save_dir}'")