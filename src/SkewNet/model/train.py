import torch
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch import optim
from model.rotated_images_dataset import RotatedImageDataset
from model.rotation_net import RotationNet
from model.train import train, test
from utils.logging_utils import setup_logging
import logging



def circular_mse(y_pred, y_true):
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



def train(model, train_loder, optimizer, device, *args):
    model.train()

    total_loss = 0.0
    for data, target in tqdm(train_loder, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = circular_mse(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loder)
    # TODO remove print statement
    print(f"Train loss: {average_loss}")
    return average_loss


def test(model, test_loader, device, *args):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = circular_mse(output, target)

            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    # TODO remove print statement
    print(f"Test loss: {average_loss}")
    return average_loss


def validation(model, validation_loader, device, *args):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(validation_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = circular_mse(output, target)

            total_loss += loss.item()

    average_loss = total_loss / len(validation_loader)
    # TODO remove print statement
    print(f"Validation loss: {average_loss}")
    return average_loss


def main():
    # setup_logging("train_model", log_level=logging.INFO)
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, prefetch_factor=4)  # No need to shuffle the test dataset

    model = RotationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        logger.info(f"Training loss: {train_loss:.4f}")

        test_loss = test(model, test_loader, device)
        logger.info(f"Test loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "/scratch/gpfs/RUSTOW/deskewing_models/rotation_net_mobilenet_v3_large_test.pth")
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main()
