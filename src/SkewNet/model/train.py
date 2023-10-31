import os
import time

import psutil
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from SkewNet.model.rotated_images_dataset import RotatedImageDataset
from SkewNet.model.rotation_net import RotationNetSmallNetworkTest


class Trainer:
    """A class for training a model.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    device : torch.device
        The device to use for training.
    writer : torch.utils.tensorboard.SummaryWriter
        The writer to use for writing TensorBoard logs.
    """

    def __init__(self, model, optimizer, device, log_dir="logs"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        subfolder = f"{model.module.__class__.__name__}_{time.strftime('%Y%m%d%H%M%S')}"
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, subfolder))

    def circular_mse(self, y_pred, y_true, scale=1.0):
        """Compute the circular mean squared error between two tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predictions from the model. Expected to be a tensor of shape
            (batch_size, 1).
        y_true : torch.Tensor
            The ground truth values. Expected to be a tensor of shape
            (batch_size, 1).
        scale : float, optional
            The scale to use for the error. This is used to scale the error
            before taking the mean.

        Returns
        -------
        torch.Tensor
            The circular mean squared error between the two tensors.
        """
        error = torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true))
        error = error * scale
        return torch.mean(error**2)

    def mse(self, y_pred, y_true):
        """Compute the mean squared error between two tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predictions from the model. Expected to be a tensor of shape
            (batch_size, 1).
        y_true : torch.Tensor
            The ground truth values. Expected to be a tensor of shape
            (batch_size, 1).

        Returns
        -------
        torch.Tensor
            The mean squared error between the two tensors.
        """
        return torch.mean((y_pred - y_true) ** 2)

    def train_epoch(self, train_loader, epoch):
        """Train the model for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader for the training dataset.
        epoch : int
            The current epoch.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_batches = len(train_loader)

        for batch_num, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.mse(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            global_step = epoch * total_batches + batch_num
            self.writer.add_scalar("Loss/train_batch", loss.item(), global_step)

            if batch_num % 100 == 0:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f"Params/{name}", param, global_step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

                self.log_system_metrics("System/Training", global_step)

        total_average_loss = total_loss / total_batches
        self.writer.add_scalar("Loss/train", total_average_loss, epoch)

        return total_average_loss

    def evaluate(self, loader, desc="test", epoch=None):
        """Evaluate the model on the given dataset.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader for the dataset.
        desc : str, optional
            The type of dataset being evaluated (e.g., "test", "validation").

        Returns
        -------
        float
            The average loss for the dataset.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = len(loader)

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.mse(output, target)
                total_loss += loss.item()

                if epoch is not None:
                    global_step = epoch * total_batches + batch_num
                    self.writer.add_scalar(f"Loss/{desc}_batch", loss.item(), global_step)

        total_average_loss = total_loss / total_batches
        if epoch is not None:
            self.writer.add_scalar(f"Loss/{desc}", total_average_loss, epoch)
            self.log_system_metrics(f"System/{desc}", epoch)

        return total_average_loss

    def setup_data_loaders(self, img_dir, annotations_file, batch_size, rank, word_size):
        """Setup the data loaders for the training and test datasets.

        Parameters
        ----------
        img_dir : str
            The path to the directory containing the images.
        annotations_file : str
            The path to the CSV file containing the annotations.
        batch_size : int
            The batch size to use for the data loaders.
        rank : int
            The rank of the current process. The rank is used to determine
            which subset of the dataset to use.
        word_size : int
            The number of processes.

        Returns
        -------
        torch.utils.data.DataLoader
            The data loader for the training dataset.
        torch.utils.data.DataLoader
            The data loader for the test dataset.
        torch.utils.data.DataLoader
            The data loader for the validation dataset.
        torch.utils.data.Sampler
            The sampler for the training dataset.
        torch.utils.data.Sampler
            The sampler for the test dataset.
        torch.utils.data.Sampler
            The sampler for the validation dataset.
        """
        # num_workers = cpu_count()
        num_workers = 32
        train_dataset = RotatedImageDataset(
            annotations_file, img_dir, subset="train", transform=self.model.module.train_transform
        )
        train_sampler = DistributedSampler(train_dataset, num_replicas=word_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=train_sampler,
            prefetch_factor=1,
        )

        test_dataset = RotatedImageDataset(
            annotations_file, img_dir, subset="test", transform=self.model.module.evaluation_transform
        )
        test_sampler = DistributedSampler(test_dataset, num_replicas=word_size, rank=rank, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=test_sampler,
            prefetch_factor=1,
        )

        validation_dataset = RotatedImageDataset(
            annotations_file, img_dir, subset="val", transform=self.model.module.evaluation_transform
        )
        validation_sampler = DistributedSampler(validation_dataset, num_replicas=word_size, rank=rank, shuffle=False)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=validation_sampler,
            prefetch_factor=1,
        )

        return train_loader, test_loader, train_sampler, test_sampler, validation_loader, validation_sampler

    def save_checkpoint(self, epoch, loss, filepath):
        """Save a checkpoint for the model.

        Parameters
        ----------
        epoch : int
            The current epoch.
        loss : float
            The current loss.
        filepath : str
            The path to save the checkpoint to.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            filepath,
        )

    def train_model(self, img_dir, annotations_file, batch_size, num_epochs, rank, word_size):
        """Train the model.

        Parameters
        ----------
        img_dir : str
            The path to the directory containing the images.
        annotations_file : str
            The path to the CSV file containing the annotations.
        batch_size : int
            The batch size to use for the data loaders.
        num_epochs : int
            The number of epochs to train the model for.
        rank : int
            The rank of the current process. The rank is used to determine
            which subset of the dataset to use.
        word_size : int
            The number of processes.
        """
        (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            validation_loader,
            validation_sampler,
        ) = self.setup_data_loaders(img_dir, annotations_file, batch_size, rank, word_size)

        model_name = f"{self.model.module.__class__.__name__}_{time.strftime('%Y%m%d%H%M%S')}.pth"

        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader, epoch)
            test_loss = self.evaluate(test_loader, desc="test", epoch=epoch)

            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"Learning_Rate/group_{i}", param_group["lr"], epoch)

            if rank == 0:
                if test_loss < best_loss:
                    best_loss = test_loss
                    checkpoint_name = f"{model_name}_best_checkpoint.pth"
                    self.save_checkpoint(epoch, test_loss, f"/scratch/gpfs/RUSTOW/deskewing_models/{checkpoint_name}")

        if rank == 0:
            torch.save(self.model.module.state_dict(), f"/scratch/gpfs/RUSTOW/deskewing_models/{model_name}")
        
        # empty the cache to avoid memory leaks and OOM errors
        torch.cuda.empty_cache()

        validation_loss = self.evaluate(validation_loader, desc="validation")

        self.writer.add_hparams(
            {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            },
            {
                "hparam/validation_loss": validation_loss,
            },
        )

        self.writer.flush()
        self.writer.close()

    def log_system_metrics(self, prefix, global_step):
        """Log system metrics to TensorBoard such as CPU and memory usage."""
        cpu_percent = psutil.cpu_percent()
        self.writer.add_scalar(f"{prefix}/CPU_Percent", cpu_percent, global_step)

        memory = psutil.virtual_memory()
        self.writer.add_scalar(f"{prefix}/Memory_Used", memory.used, global_step)


def main(rank, word_size):
    """The main function for training a model.

    Parameters
    ----------
    rank : int
        The rank of the current process. The rank is used to determine
        which subset of the dataset to use.
    word_size : int
        The number of processes.
    """
    img_dir = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data"
    annotations_file = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_image_angles.csv"
    log_dir = "/scratch/gpfs/eh0560/SkewNet/logs/tensorboard"

    torch.cuda.set_device(rank)

    batch_size = 48
    learning_rate = 0.004
    num_epochs = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group("nccl", rank=rank, world_size=word_size)
    model = RotationNetSmallNetworkTest().to(device)
    # model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, device, log_dir)
    trainer.train_model(img_dir, annotations_file, batch_size, num_epochs, rank, word_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    word_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(word_size,), nprocs=word_size, join=True)
