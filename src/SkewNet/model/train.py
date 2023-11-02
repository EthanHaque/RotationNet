import os

import pytorch_lightning as pl
import torch
import torch.optim as optim

from SkewNet.model.rotated_images_dataset import RotatedImageDataset
from SkewNet.model.rotation_net import ModelRegistry
from SkewNet.model.callbacks.better_progress_bar import BetterProgressBar
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


class LightningRotationNet(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def mse(self, y_pred, y_true, scale=1):
        return torch.mean((y_pred - y_true) ** 2) * scale
    
    def mae(self, y_pred, y_true, scale=1):
        return torch.mean(torch.abs(y_pred - y_true)) * scale

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_mae", self.mae(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def _make_grid(self, x, y, y_hat):
        num_images = min(x.size(0), 4)  
        x = x[:num_images]

        degrees = y_hat[:num_images] * 180 / 3.141592
        degrees = degrees.view(-1).tolist()  

        rotated_images = []
        for i in range(num_images):  
            rotated_image = transforms.functional.rotate(x[i].unsqueeze(0), angle=-degrees[i], expand=False)
            rotated_images.append(rotated_image.squeeze(0))  # Remove the batch dimension after rotating

        # Create a grid of images with 2 images per row
        grid = make_grid(rotated_images, nrow=2)
        return grid
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_mae", self.mae(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.log("test_mae", self.mae(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]


def setup_data_loaders(annotations_file, img_dir, batch_size, num_workers, prefetch_factor, train_transform, val_transform, test_transform):
    train_dataset = RotatedImageDataset(
        annotations_file,
        img_dir,
        subset="train",
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    test_dataset = RotatedImageDataset(
        annotations_file,
        img_dir,
        subset="test",
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    val_dataset = RotatedImageDataset(
        annotations_file,
        img_dir,
        subset="val",
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader, test_loader



def main():
    BATCH_SIZE = 48
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 25
    NUM_WORKERS = 12
    NUM_NODES = 1
    ALLOCATED_GPUS_PER_NODE = 2
    

    # TODO: test the impact of using float16 instead of float32
    torch.set_float32_matmul_precision("medium")

    model = ModelRegistry.get_model("HugeTestNetwork")
    model = LightningRotationNet(model, LEARNING_RATE, WEIGHT_DECAY)

    train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),])
    test_transform = transforms.Compose([])
    val_transform = transforms.Compose([])

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=ALLOCATED_GPUS_PER_NODE,
        num_nodes=NUM_NODES,
        max_epochs=NUM_EPOCHS,
        logger=pl.loggers.TensorBoardLogger("logs/tensorboard", name="SkewNet"),
        callbacks=[
            BetterProgressBar(),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath="/scratch/gpfs/RUSTOW/deskewing_models",
                filename="SkewNet-{epoch:03d}-{val_loss:.5f}",
                monitor="val_loss",
            )
        ],
    )

    img_dir = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data"
    annotations_file = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_image_angles.csv"
    
    train_loader, val_loader, test_loader = setup_data_loaders(
        annotations_file,
        img_dir,
        BATCH_SIZE,
        NUM_WORKERS,
        prefetch_factor=1,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
