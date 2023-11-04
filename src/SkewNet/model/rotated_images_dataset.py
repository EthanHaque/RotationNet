import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from dataclasses import dataclass

@dataclass
class DataConfig:
    annotations_file: str
    img_dir: str
    truncate: float = 1.0


class RotatedImageDataset(Dataset):
    def __init__(self, data_config, subset, transform=None, target_transform=None):
        annotations_file = data_config.annotations_file
        img_dir = data_config.img_dir
        truncate = data_config.truncate

        if subset not in ["train", "val", "test"]:
            raise ValueError("subset must be one of [train, val, test]")
        if truncate < 0.0 or truncate > 1.0:
            raise ValueError("truncate must be between 0.0 and 1.0")
        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels["split"] == subset]
        number_samples = int(len(self.img_labels) * truncate)
        self.img_labels = self.img_labels[:number_samples]
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
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

