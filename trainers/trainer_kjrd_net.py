import time, torch
import torch.nn as nn
from models.KJRDNet_wo_detection import KJRDNet_wo_detection
from models.faster_rcnn import faster_rcnn
from trainers.trainer_base import Trainer

class KJRDNetTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)
        self.use_diffusion = config.train.use_diffusion
        self.main_block=self.init_main_block()
        self.detector=self.init_detector()
        
        param_main_block=[p for p in self.main_block.parameters() if p.requires.grad]
        param_detector=[p for p in self.detector.parameters() if p.requires.grad]
        self.optimizer = torch.optim.SGD([
            {'params':param_main_block,'lr':self.lr*0.02},
            {'params':param_detector,'lr':self.lr}
        ], momentum=0.9, weight_decay=1e-4)
        
        self.loss_lambda = config.loss.lmbda
        self.detector_criterion=DetectionLoss()

    def init_main_block(self):
        model = KJRDNet_wo_detection(
            ffa_weights='./output_models/ffa_net_dotah_ffa_net.pth',
            RCAN_weights='./output_models/rcan_dotah_rcan_1.pth',
            VIT_weights=None,
            diffusion_weights='./output_models/diffusion_net_dotah_ffa_net.pth',
            use_diffusion=self.use_diffusion
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

            for i, (hazy_images, targets) in enumerate(self.train_loader):  #need to update train loader for this
                print('Training batch:', i)
                hazy_images = [img.to(self.device) for img in hazy_images]
                ref_images = [t['ref_image'].to(self.device) for t in targets]
                boxes = [t['object_labels']['boxes'].to(self.device) for t in targets]
                labels = [t['object_labels']['labels'].to(self.device) for t in targets]

                

                # Forward pass
                fused_images = self.main_block(hazy_images)
                detector_output=self.detector(fused_images,[boxes,labels])

                loss_reconstruction = nn.L1Loss()(fused_images, ref_images)
                loss_detector=self.detector_criterion(detector_output)
                loss=loss_detector + loss_reconstruction*self.loss_lambda

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_loss = self.validate(epoch)

        self.save_model(self.main_block, f'{self.output_dir}/KJRDnet_main_block_dataset_{self.dataset.lower()}.pth')
        self.save_model(self.detector, f'{self.output_dir}/KJRDnet_detector_dataset_{self.dataset.lower()}.pth')
        print(f"Completed in {(time.time()-start):.3f}.")

    
class DetectionLoss:
    def __call__(self, detector_output):
        return sum(loss for loss in detector_output.values())
        