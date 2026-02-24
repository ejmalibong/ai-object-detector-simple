# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T

# Mapping class names to integer labels
CLASS_TO_IDX = {"book": 0, "cup": 1, "bottle": 2}

class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # list of all image filenames
        self.img_files = [
            f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, img_name.replace(".jpg", ".xml"))

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Parse XML
        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_TO_IDX:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target

# Example transform for training
def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)