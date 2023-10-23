import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class RotatedImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, subset, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels["split"] == subset]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).type(torch.FloatTensor) / 255.0
        angle = self.img_labels.iloc[idx, 1]
        label = torch.tensor([angle], dtype=torch.float32)
        label = torch.deg2rad(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
