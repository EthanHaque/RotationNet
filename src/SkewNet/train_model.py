import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from model.rotated_images_dataset import RotatedImageDataset
from model.rotation_net import RotationNet
from model.train import train, test
from utils.logging_utils import setup_logging
import logging


def main():
    # setup_logging("train_model", log_level=logging.INFO)
    logger = logging.getLogger(__name__)

    img_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"
    annotations_file = "/scratch/gpfs/RUSTOW/deskewing_datasets/synthetic_image_angles.csv"

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    train_dataset = RotatedImageDataset(annotations_file, img_dir, subset="train", transform=image_transforms["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = RotatedImageDataset(annotations_file, img_dir, subset="test", transform=image_transforms["test"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle the test dataset

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
