import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.transforms.v2 as transformsV2
from augmentations import ManualRandomHorizontalFlip
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import VOCDataset
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

lr = 1e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
weight_decay = 0.0005 # As seen in paper
epochs = 1 # Paper used 135 epochs
num_workers = 4
pin_memory = True
load_model = False
load_model_file = 'yolov1_weights_7b7.pt'
img_dir = 'PascalVOC/images'
label_dir = 'PascalVOC/labels'

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        flipped = False
        for t in self.transforms:
            try:
                # Try to call with both img and boxes
                result = t(img, bboxes)
                if isinstance(result, tuple) and len(result) == 3:
                    img, bboxes, was_flipped = result
                    flipped |= was_flipped
                else:
                    img, bboxes = result
            except TypeError:
                # Fallback: assume transform only wants img
                img = t(img)
        return img, bboxes, flipped
    
# Validation transformation
transform_v = Compose([
    transformsV2.ToImage(),
    transformsV2.ToDtype(torch.float32,scale=True),
    transformsV2.Resize((448, 448)),])

def val_fn(val_loader, model, loss_fn):
    model.eval()
    mean_loss = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())
    model.train()
    return sum(mean_loss) / len(mean_loss)

def main():
    model = YoloV1(split_size=7,num_boxes=2,num_classes=20).to(device)

    model.load_state_dict(torch.load('yolov1_weights_7b7.pt',map_location=device))

    test_dataset = VOCDataset(
        'PascalVOC/test.csv',
        transform=transform_v,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(epochs):
        print(f'\nepoch: {epoch+1}')

        # mAP score for validation set
        pred_boxes_v,target_boxes_v = get_bboxes(
            test_loader,model,iou_threshold=0.3,threshold=0.4
        )   
        mean_avg_prec_v = mean_average_precision(
            pred_boxes_v,target_boxes_v,iou_threshold=0.3,box_format='midpoint'
        )

        print(f"Val mAP: {mean_avg_prec_v:.4f}")

if __name__ == '__main__':
    main()