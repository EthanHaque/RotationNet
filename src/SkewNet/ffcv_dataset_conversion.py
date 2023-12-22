from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, FloatField
from SkewNet.model.rotated_images_dataset import DataConfig, RotatedImageDataset
import numpy as np

def setup_data_loaders():
    data_config = DataConfig(
        annotations_file= "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data_angles.csv",
        img_dir= "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data/",
        truncate= 1.0,
        min_angle=-2 * np.pi,
        max_angle=2 * np.pi
    )
    train_dataset = RotatedImageDataset(data_config, subset="train")
    val_dataset = RotatedImageDataset(data_config, subset="val")
    test_dataset = RotatedImageDataset(data_config, subset="test")

    return train_dataset, val_dataset, test_dataset

def main():
    train_dataset, val_dataset, test_dataset = setup_data_loaders()
    write_path = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data_ffcv/test"

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        "image": RGBImageField(jpeg_quality=95),
        "document_angle": FloatField()
    })

    # Write dataset
    writer.from_indexed_dataset(test_dataset)


if __name__ == "__main__":
    main()