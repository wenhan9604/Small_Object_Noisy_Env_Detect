import time, torch
import torch.nn as nn
from models.KJRDNet_wo_detection import KJRDNet_wo_detection
from models.faster_rcnn import faster_rcnn
from trainers.trainer_base import Trainer
from torchvision import transforms
from models.faster_rcnn_loss import eval_forward

class KJRDNetTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        self.use_diffusion = config.train.use_diffusion
        self.main_block=self.init_main_block()
        self.detector=self.init_detector()
        
        param_main_block=[p for p in self.main_block.parameters() if p.requires_grad]
        param_detector=[p for p in self.detector.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD([
            {'params':param_main_block,'lr':self.lr*0.02},
            {'params':param_detector,'lr':self.lr}
        ], momentum=0.9, weight_decay=1e-4)
        
        self.loss_lambda = config.loss.lmbda
        self.detector_criterion=DetectionLoss()

    def init_main_block(self):
        model = KJRDNet_wo_detection(
            ffa_weights='./output_models/ffa_net_dotah_ffa_net.pth',
            RCAN_weights='./output_models/rcan_dotah_rcan_2.pth',
            VIT_weights= './output_models/mae_pretrain_vit_large.pth',
            diffusion_weights='./output_models/diffusion_net_dotah_ffa_net.pth',
            use_diffusion=self.use_diffusion,
            device=self.device
        ).to(self.device)
        return model
    
    def init_detector(self):
        return faster_rcnn().to(self.device)

    def train(self):
        print(f"Training KJRD-Net")
        # self.main_model.train()
        start = time.time()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.main_block.train()
            self.detector.train()
            transform = transforms.ToTensor()

            for i, (hazy_images, targets) in enumerate(self.train_loader):  #need to update train loader for this
                print('Training batch:', i)
                hazy_images = [transform(img) for img in hazy_images]
                # hazy_image = transform(hazy_images[i]).to(self.device)
                hazy_images = torch.stack(hazy_images, dim=0).to(self.device)
                # ref_images = [t['ref_image'].to(self.device) for t in targets]
                # ref_images = targets[i]['ref_image'].to(self.device)
                ref_images = [t['ref_image'] for t in targets]
                ref_images = torch.stack(ref_images, dim=0).to(self.device)

                # print(f"hazy_images shape: {hazy_images.shape}")

                # Forward pass
                fused_images = self.main_block(hazy_images)

                # boxes_and_labels = [{'boxes':t['object_labels']['boxes'].to(self.device),'labels':t['object_labels']['labels'].to(self.device)} for t in targets]
                
                boxes_and_labels = [    
                    {"boxes": t['object_labels']['boxes'].to(self.device), "labels": t['object_labels']['labels'].to(self.device)} 
                    if len(t['object_labels']['boxes']) > 0  # Check if boxes exist    
                    else {"boxes": torch.zeros((0, 4), dtype=torch.int64, device=self.device), "labels": torch.zeros(0, dtype=torch.int64, device=self.device)}
                    for t in targets]

                # print(f"fused_images type: {fused_images.shape}")
                # print(f"boxes_and_labels: {boxes_and_labels}")

                detector_output=self.detector(list(fused_images),boxes_and_labels)

                loss_reconstruction = nn.L1Loss()(fused_images, ref_images)
                loss_detector=self.detector_criterion(detector_output)
                loss=loss_detector + loss_reconstruction*self.loss_lambda

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                torch.cuda.empty_cache()

                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")

            # if hasattr(self, 'val_loader') and self.val_loader is not None:
                # val_loss = self.validate(epoch)
                # val_loss, detections = eval_forward(self.detector.detector, list(fused_images),boxes_and_labels)
                # avg_val_loss = val_loss / len(self.val_loader)
                # print(f"Epoch [{epoch+1}] Validation Loss: {val_loss}, , Val_loader_length {len(self.val_loader)}")

        self.save_model(self.main_block, f'{self.output_dir}/KJRDnet_main_block_dataset_{self.config.network.model.lower()}.pth')
        self.save_model(self.detector, f'{self.output_dir}/KJRDnet_detector_dataset_{self.config.network.model.lower()}.pth')
        print(f"Completed in {(time.time()-start):.3f}.")

    def validate(self, epoch):
        # self.model.eval()
        self.main_block.eval()
        self.detector.eval()
        val_loss = 0.0
        transform = transforms.ToTensor()

        with torch.no_grad():
            for inputs, targets in self.val_loader:

                input_imgs = [transform(img) for img in inputs]
                # hazy_image = transform(hazy_images[i]).to(self.device)
                input_imgs = torch.stack(input_imgs, dim=0).to(self.device)
                # ref_images = [t['ref_image'].to(self.device) for t in targets]
                # ref_images = targets[i]['ref_image'].to(self.device)
                ref_images = [t['ref_image'] for t in targets]
                ref_images = torch.stack(ref_images, dim=0).to(self.device)

                # print(f"hazy_images shape: {hazy_images.shape}")

                # Forward pass
                fused_images = self.main_block(input_imgs)

                # boxes_and_labels = [{'boxes':t['object_labels']['boxes'].to(self.device),'labels':t['object_labels']['labels'].to(self.device)} for t in targets]
                
                boxes_and_labels = [    
                    {"boxes": t['object_labels']['boxes'].to(self.device), "labels": t['object_labels']['labels'].to(self.device)} 
                    if len(t['object_labels']['boxes']) > 0  # Check if boxes exist    
                    else {"boxes": torch.zeros((0, 4), dtype=torch.int64, device=self.device), "labels": torch.zeros(0, dtype=torch.int64, device=self.device)}
                    for t in targets]

                # print(f"fused_images type: {fused_images.shape}")
                # print(f"boxes_and_labels: {boxes_and_labels}")

                detector_output=self.detector(list(fused_images),boxes_and_labels)

                loss_reconstruction = nn.L1Loss()(fused_images, ref_images)
                loss_detector=self.detector_criterion(detector_output)
                loss=loss_detector + loss_reconstruction*self.loss_lambda

                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Epoch [{epoch+1}] Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

class DetectionLoss:
    def __call__(self, detector_output):
        return sum(loss for loss in detector_output.values())
        