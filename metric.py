import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm
 
from models.KJRDNet_wo_detection import KJRDNet_wo_detection
from models.faster_rcnn import faster_rcnn
from data_source.kjrd_dataset import KJRDDataset
from torchvision import transforms
 
def evaluate_object_detection_multiclass(test_dataset, iou_threshold=0.1):
    total_TP, total_FP, total_FN = 0, 0, 0


    for sample in test_dataset:
        gt_boxes = sample['gt_boxes']
        gt_classes = sample['gt_classes']
        pred_boxes = sample['pred_boxes']
        pred_classes = sample['pred_classes']
        pred_scores = sample['pred_scores']
 
        if len(pred_boxes) == 0:
            total_FN += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_FP += len(pred_boxes)
            continue
 
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_idx]
        pred_classes = pred_classes[sorted_idx]
 
        print(f"pred: {pred_boxes[0]}, gt: {gt_boxes[0]}")
        print(f"pred_max: {pred_boxes[0].max()}, gt_max{gt_boxes[0].max()}")

        matched_gt = set()
        ious = box_iou(pred_boxes, gt_boxes)
 
        for pred_idx in range(len(pred_boxes)):
            best_iou, best_gt_idx = ious[pred_idx].max(0)
            if (best_iou >= iou_threshold and
                best_gt_idx.item() not in matched_gt):
                # best_gt_idx.item() not in matched_gt and
                # pred_classes[pred_idx] == gt_classes[best_gt_idx]):
                total_TP += 1
                matched_gt.add(best_gt_idx.item())
            else:
                total_FP += 1
 
        total_FN += (len(gt_boxes) - len(matched_gt))
 
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
 
    return precision, recall, f1
 
def run_inference(model_main, model_detector, dataset, device, max_samples=30):
    model_main.eval()
    model_detector.eval()

    loader = DataLoader(
        Subset(dataset, range(max_samples)),
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn  # <--- use this
    )
    results = []
    transform = transforms.ToTensor()
 
    with torch.no_grad():
        for i, (hazy_images, targets) in enumerate(tqdm(loader, desc="Running Inference")):
            print('Inference batch:', i)
            hazy_images = [transform(img).to(device) for img in hazy_images]  # Apply transform to each image
            hazy_images = torch.stack(hazy_images, dim=0)
 
            fused_images = model_main(hazy_images)
 
            boxes_and_labels = [
                {"boxes": t['object_labels']['boxes'].to(device), "labels": t['object_labels']['labels'].to(device)}
                if len(t['object_labels']['boxes']) > 0
                else {"boxes": torch.zeros((0, 4), dtype=torch.int64, device=device), "labels": torch.zeros(0, dtype=torch.int64, device=device)}
                for t in targets
            ]
 
            outputs = model_detector(list(fused_images), boxes_and_labels)
 
            for idx in range(len(targets)):
                results.append({
                    'gt_boxes': boxes_and_labels[idx]['boxes'].cpu(),
                    'gt_classes': boxes_and_labels[idx]['labels'].cpu(),
                    'pred_boxes': outputs[idx]['boxes'].cpu(),
                    'pred_classes': outputs[idx]['labels'].cpu(),
                    'pred_scores': outputs[idx]['scores'].cpu()
                })
 
    return results

def custom_collate_fn(batch):

    hazy_images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return hazy_images, targets 
 
def make_split(split, transform):
    # label_path = f'./raw_data/dota_hazed/val/labelTxt' 
    # image_path = f'./raw_data/dota_hazed/val/images'
    label_path = os.path.join(os.getcwd(), 'raw_data', 'dota_hazed', split, 'labelTxt')
    hazy_image_path = os.path.join(os.getcwd(), 'raw_data', 'dota_hazed', split, 'images')
    clear_image_path = os.path.join(os.getcwd(), 'raw_data', 'dota_orig', split, 'images')

    # print(f"image_path: {image_path}")
    # print(f"label_path: {label_path}")
    # if not (os.path.isfile(label_path) and os.path.isdir(image_path)):
    #     raise FileNotFoundError(f"Missing label file or image directory for: {split}")

    # if not (os.path.isdir(label_path)):
    #     raise FileNotFoundError(f"Missing label file or for: {split}")

    # if not (os.path.isdir(image_path)):
    #     raise FileNotFoundError(f"Missing image directory for: {split}")

    return KJRDDataset(labels_dir=label_path, clear_dir=clear_image_path, hazy_dir=hazy_image_path, transform_clear=transform)
 
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Load models
    main_model = KJRDNet_wo_detection(
        ffa_weights=None,
        RCAN_weights=None,
        VIT_weights='./output_models/mae_pretrain_vit_large.pth',
        diffusion_weights='./output_models/diffusion_net_dotah_ffa_net.pth',
        use_diffusion=True
    ).to(device)
    
    main_model.load_state_dict(torch.load('./output_models/KJRDnet_main_block_dataset_kjrd_diffusion.pth', map_location=device))
 
    detector = faster_rcnn().to(device)
    detector.load_state_dict(torch.load('./output_models/KJRDnet_detector_dataset_kjrd_diffusion.pth', map_location=device))
 
    # Load dataset
    base_dimension=256
    transform_clear = transforms.Compose([
        transforms.Resize((base_dimension*2,base_dimension*2)),
        transforms.ToTensor()
    ])

    test_dataset = make_split(split='val', transform=transform_clear)
 
    # Run inference and evaluate
    predictions = run_inference(main_model, detector, test_dataset, device)
    # print("Results\n")
    # print(predictions)
    precision, recall, f1 = evaluate_object_detection_multiclass(predictions)
 
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")