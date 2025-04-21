import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.transforms.v2 as transformsV2
from augmentations import ManualRandomHorizontalFlip, YoloRandomAffine
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

seed = 123
torch.manual_seed(seed)

# Hyperparameters #
# Paper uses schedule: Slowly rises from 1e-3 to 1e-2 (in unspecified number of epochs; assumed 10),
# 1e-2 until epoch 75, then 1e-3 for 30 epochs, and finally 1e-4 for 30 epochs
lr = 1e-5 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
weight_decay = 0.0005 # As seen in paper
epochs = 135 # Paper used 135 epochs
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
            # Check if transform needs both image and boxes
            if getattr(t, 'requires_boxes', False):
                result = t(img, bboxes)
            else:
                result = t(img)

            # Handle returned values
            if isinstance(result, tuple):
                if len(result) == 3:
                    img, bboxes, was_flipped = result
                    flipped |= was_flipped
                elif len(result) == 2:
                    img, bboxes = result
                else:
                    raise ValueError(f"Unexpected return format from {t}: {result}")
            else:
                img = result  # torchvision-style image-only transforms

        return img, bboxes, flipped

# Training transformation
transform_t = Compose([
    transformsV2.ToImage(),
    transformsV2.ToDtype(torch.float32, scale=True),
    YoloRandomAffine(),
    ManualRandomHorizontalFlip(p=0.5),
    transformsV2.ColorJitter(brightness=1.5, saturation=1.5),
    transformsV2.Resize((448, 448)),])

# Validation transformation
transform_v = Compose([
    transformsV2.ToImage(),
    transformsV2.ToDtype(torch.float32,scale=True),
    transformsV2.Resize((448, 448)),])

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def train_fn(train_loader,model, optimizer,loss_fn):
    loop = tqdm(train_loader,leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    return sum(mean_loss)/len(mean_loss)

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
    lr = 1e-5

    model = YoloV1(split_size=7,num_boxes=2,num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr,weight_decay=weight_decay
    )
    loss_fn = YoloLoss()

    if load_model:
        model.load_state_dict(torch.load('yolov1_weights_7b7.pt',map_location=device))

    train_dataset = VOCDataset(
        'PascalVOC/train.csv',
        transform=transform_t,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    test_dataset = VOCDataset(
        'PascalVOC/test.csv',
        transform=transform_v,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    train_losses = []
    val_losses = []
    map_scores_t = []
    map_scores_v = []

    for epoch in range(epochs):
        # Learning rate schedule
        if (epoch+1) <= 30:
            lr = 1e-5 + (3.33e-5)*(epoch+1)
        elif (epoch+1) > 30 & (epoch+1) <= 75:
            lr = 1e-3
        elif (epoch+1) > 75 & (epoch+1) <= 105:
            lr = 1e-4
        else:
            lr = 1e-5

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print(f'\nepoch: {epoch+1}')

        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss = val_fn(test_loader, model, loss_fn)

        # mAP score for training set
        pred_boxes,target_boxes = get_bboxes(
            train_loader,model,iou_threshold=0.5,threshold=0.4
        )   
        mean_avg_prec = mean_average_precision(
            pred_boxes,target_boxes,iou_threshold=0.5,box_format='midpoint'
        )

        # mAP score for validation set
        pred_boxes_v,target_boxes_v = get_bboxes(
            test_loader,model,iou_threshold=0.5,threshold=0.4
        )   
        mean_avg_prec_v = mean_average_precision(
            pred_boxes_v,target_boxes_v,iou_threshold=0.5,box_format='midpoint'
        )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train mAP: {mean_avg_prec:.4f} | Val mAP: {mean_avg_prec_v:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        map_scores_t.append(mean_avg_prec)
        map_scores_v.append(mean_avg_prec_v)

        
        train_fn(train_loader,model,optimizer,loss_fn)
        if (epoch+1) % 10:
            torch.save(model.state_dict(),'yolov1_weights_7b7.pt')
        elif (epoch+1) == 135:
            torch.save(model.state_dict(),'last_7b7.pt')

    max_loss = 5000
    train_losses_clipped = [min(loss, max_loss) for loss in train_losses]
    val_losses_clipped = [min(loss, max_loss) for loss in val_losses]
    train_smooth = moving_average(train_losses_clipped, window_size=10)
    val_smooth = moving_average(val_losses_clipped, window_size=10) 

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")

    # Clipped and smoothed plot
    plt.figure()
    plt.plot(train_smooth, label="Training Loss")
    plt.plot(val_smooth, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot_smooth.png")

    plt.figure()
    plt.plot(map_scores_t, label="Training mAP")
    plt.plot(map_scores_v,label="Validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.savefig("map_plot.png")

        

if __name__ == '__main__':
    main()