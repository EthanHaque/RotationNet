import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from model.rotated_images_dataset import RotatedImageDataset
from model.rotation_net import RotationNetMobileNetV3Backbone
from utils.logging_utils import setup_logging
import logging


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
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(loader, desc=desc, leave=False):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.circular_mse(output, target)
                total_loss += loss.item()
        return total_loss / len(loader)


def main():
    setup_logging("train_model", log_level=logging.INFO)
    logger = logging.getLogger(__name__)

    img_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"
    annotations_file = "/scratch/gpfs/RUSTOW/deskewing_datasets/synthetic_image_angles.csv"

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    train_dataset = RotatedImageDataset(annotations_file, img_dir, subset="train", transform=image_transforms["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, prefetch_factor=4)

    test_dataset = RotatedImageDataset(annotations_file, img_dir, subset="test", transform=image_transforms["test"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, prefetch_factor=4) 

    model = RotationNetMobileNetV3Backbone().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, optimizer, device)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Training loss: {train_loss:.4f}")

        test_loss = trainer.evaluate(test_loader, desc="Testing")
        logger.info(f"Test loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "/path/to/save/model.pth")
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main()
