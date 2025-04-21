import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.v2 import RandomAffine
from torchvision.tv_tensors import BoundingBoxes
from utils import plot_image_with_boxes
import torch
import sys
import numpy as np
from PIL import Image, ImageEnhance

class ManualRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.requires_boxes = True

    def __call__(self, img, boxes):
        if random.random() < self.p:
            img = F.hflip(img)
            return img, boxes, True
        else:
            return img, boxes, False

class YoloRandomAffine:
    def __init__(self, degrees=0.0, translate=(0.2, 0.2), scale=(0.8, 1.2)):
        self.affine = RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=0.0
        )
        self.requires_boxes = True
        

    def __call__(self, image, boxes):
        """
        boxes: Tensor [N, 5] where columns are [cls, cx, cy, w, h] in YOLO format
        """
        if boxes.shape[0] == 0:
            return self.affine(image), boxes  # nothing to transform

        img_h, img_w = image.shape[-2:]  # (C, H, W) layout

        labels = boxes[:, 0]
        box_coords = boxes[:, 1:].clone()

        # Convert YOLO → pixel space
        box_coords[:, 0] *= img_w  # cx
        box_coords[:, 1] *= img_h  # cy
        box_coords[:, 2] *= img_w  # w
        box_coords[:, 3] *= img_h  # h

        tv_boxes = BoundingBoxes(box_coords, format="CXCYWH", canvas_size=(img_h, img_w))

        # Apply affine transform
        image, tv_boxes = self.affine(image, tv_boxes)

        # Convert back to YOLO format
        box_coords = tv_boxes.data

        box_coords[:, 0] /= img_w
        box_coords[:, 1] /= img_h
        box_coords[:, 2] /= img_w
        box_coords[:, 3] /= img_h

        boxes_out = torch.cat([labels.unsqueeze(1), box_coords], dim=1)
        return image, boxes_out
