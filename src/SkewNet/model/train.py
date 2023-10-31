import os

import lightning.pytorch as pl
import torch
import torch.optim as optim

from SkewNet.model.rotated_images_dataset import RotatedImagesModule
from SkewNet.model.rotation_net import RotationNetSmallNetworkTest


class LightningRotationNet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def mse(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse(y_hat, y)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.004)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]


def main():
    BATCH_SIZE = 48
    NUM_EPOCHS = 3
    NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
    NUM_NODES = int(os.environ["SLURM_NNODES"])
    ALLOCATED_GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])

    # Trades off precision for performance
    torch.set_float32_matmul_precision("medium" | "high")

    model = LightningRotationNet(RotationNetSmallNetworkTest())

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=ALLOCATED_GPUS_PER_NODE,
        num_nodes=NUM_NODES,
        max_epochs=NUM_EPOCHS,
        logger=pl.loggers.TensorBoardLogger("logs/tensorboard", name="SkewNet"),
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    )

    img_dir = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data"
    annotations_file = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_image_angles.csv"
    images = RotatedImagesModule(
        img_dir,
        annotations_file,
        BATCH_SIZE,
        NUM_WORKERS,
    )

    trainer.fit(model, images)


if __name__ == "__main__":
    main()
