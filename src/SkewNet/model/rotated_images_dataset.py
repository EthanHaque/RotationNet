import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


@dataclass
class DataConfig:
    annotations_file: str
    img_dir: str
    truncate: float = 1.0
    min_angle: float = 0.0
    max_angle: float = 2 * np.pi


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

        # normalize angles to range [0, 2pi]
        # self.img_labels["document_angle"] = self.img_labels["document_angle"] % (2 * np.pi)
        min_angle = data_config.min_angle
        max_angle = data_config.max_angle

        if min_angle > max_angle:
            self.img_labels = self.img_labels[
                (self.img_labels["document_angle"] >= min_angle) | (self.img_labels["document_angle"] <= max_angle)
            ]
        else:
            self.img_labels = self.img_labels[
                (self.img_labels["document_angle"] >= min_angle) & (self.img_labels["document_angle"] <= max_angle)
            ]

        self.angle_interval = (min_angle, max_angle)

        number_samples = int(len(self.img_labels) * truncate)
        self.img_labels = self.img_labels.sample(number_samples)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform    

    def get_angle_interval(self):
        return self.angle_interval

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
