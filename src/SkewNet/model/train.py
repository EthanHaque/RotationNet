import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch import optim
from rotated_images_dataset import RotatedImageDataset
from rotation_net import RotationNetSmallNetworkTest
from SkewNet.utils.logging_utils import setup_logging
import logging
import time


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
    """
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device


    def circular_mse(self, y_pred, y_true):
        """Compute the circular mean squared error between two tensors.

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
            The circular mean squared error between the two tensors.
        """
        error = torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true))
        return torch.mean(error ** 2)


    def train_epoch(self, train_loader):
        """Train the model for one epoch.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader for the training dataset.
            
        Returns
        -------
        float
            The average loss for the epoch.
        """
        logger = logging.getLogger(__name__)
        self.model.train()
        total_loss = 0.0
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.circular_mse(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            logger.debug(f"Batch loss: {loss.item():.4f}")
        return total_loss / len(train_loader)


    def evaluate(self, loader, desc="Testing"):
        """Evaluate the model on the given dataset.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader for the dataset.
        desc : str, optional
            The description to use for the progress bar.

        Returns
        -------
        float
            The average loss for the dataset.
        """
        logger = logging.getLogger(__name__)
        self.model.eval()
        total_loss = 0.0
        total_batches = len(loader)

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.circular_mse(output, target)
                total_loss += loss.item()

                avg_loss = total_loss / batch_num

                logger.debug(
                    "Batch %d/%d - Loss: %.4f | Cumulative Loss: %.4f | Average Loss: %.4f",
                    batch_num, total_batches, loss.item(), total_loss, avg_loss
                )

        avg_loss = total_loss / total_batches
        logger.info("%s - Average Loss: %.4f", desc, avg_loss)

        return avg_loss
    

    def setup_data_loaders(self, img_dir, annotations_file, batch_size, model):
        """Setup the data loaders for the training and test datasets.

        Parameters
        ----------
        img_dir : str
            The path to the directory containing the images.
        annotations_file : str
            The path to the CSV file containing the annotations.
        batch_size : int
            The batch size to use for the data loaders.
        model : torch.nn.Module
            The model to use for the data loaders.

        Returns
        -------
        torch.utils.data.DataLoader
            The data loader for the training dataset.
        torch.utils.data.DataLoader
            The data loader for the test dataset.
        """
        train_dataset = RotatedImageDataset(annotations_file, img_dir, subset="train", transform=model.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, prefetch_factor=4)

        test_dataset = RotatedImageDataset(annotations_file, img_dir, subset="test", transform=model.evaluation_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, prefetch_factor=4) 

        return train_loader, test_loader
    

    def train_model(self, img_dir, annotations_file, batch_size, num_epochs):
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
        """
        logger = logging.getLogger(__name__)
        train_loader, test_loader = self.setup_data_loaders(img_dir, annotations_file, batch_size, self.model)

        model_name = f"{self.model.__class__.__name__}_{time.strftime('%Y%m%d%H%M%S')}.pth"

        for epoch in trange(num_epochs, desc="Epoch"):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")

            test_loss = self.evaluate(test_loader, desc="Testing")
            logger.info(f"Test loss: {test_loss:.4f}")

        torch.save(self.model.state_dict(), f"/scratch/gpfs/RUSTOW/deskewing_models/{model_name}")
        logger.info("Model saved successfully.")


def main():
    img_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"
    annotations_file = "/scratch/gpfs/RUSTOW/deskewing_datasets/synthetic_image_angles.csv"

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = RotationNetSmallNetworkTest().to(device)

    logfile_prefix = f"train_model_{model.__class__.__name__}"
    setup_logging(logfile_prefix, log_level=logging.DEBUG, log_to_stdout=False)


    logger = logging.getLogger(__name__)
    logger.debug("Device: %s", device)
    logger.debug("Batch size: %d", batch_size)
    logger.debug("Learning rate: %.4f", learning_rate)
    logger.debug("Number of epochs: %d", num_epochs)
    logger.debug("Model: %s", model.__class__.__name__)
    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, device)
    trainer.train_model(img_dir, annotations_file, batch_size, num_epochs)


if __name__ == "__main__":
    main()
