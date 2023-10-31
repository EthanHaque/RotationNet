import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


#TODO: this isn't working 
class RotatedImagesModule(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_file, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = 1
        self.train_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ]
        )
        self.val_transform = transforms.Compose([])
        self.test_transform = transforms.Compose([])

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = RotatedImageDataset(
                self.annotations_file, self.data_dir, "train", transform=self.train_transform
            )

            self.val_dataset = RotatedImageDataset(
                self.annotations_file, self.data_dir, "val", transform=self.val_transform
            )
        if stage == "test" or stage is None:
            self.test_dataset = RotatedImageDataset(
                self.annotations_file, self.data_dir, "test", transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
