import time

from SkewNet.model.rotated_images_dataset import RotatedImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ProfilingConfig:
    """
    Configuration for profiling the data loading of a RotatedImageDataset.
    
    Parameters
    ----------
    img_dir : str
        Directory where images are stored.
    annotations_file : str
        Path to the CSV file with annotations for the images.
    subset : str
        Subset to use (e.g., 'test', 'train').
    batch_sizes : list
        List of batch sizes to profile.
    num_workers : list
        List of numbers of workers to use for DataLoader.
    num_epochs : int
        Number of epochs to profile over.

    Attributes
    ----------
    img_dir : str
        Directory where images are stored.
    annotations_file : str
        Path to the CSV file with annotations for the images.
    subset : str
        Subset to use (e.g., 'test', 'train').
    batch_sizes : list
        List of batch sizes to profile.
    num_workers : list
        List of numbers of workers to use for DataLoader.
    num_epochs : int
        Number of epochs to profile over.
    """
    def __init__(self, img_dir, annotations_file, subset, batch_sizes, num_workers, num_epochs):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.subset = subset
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.num_epochs = num_epochs


def profile_data_loading(batch_size, num_workers, dataset):
    """
    Profiles the data loading time for a given batch size and number of workers using a DataLoader.

    Parameters
    ----------
    batch_size : int
        The size of each batch of data.
    num_workers : int
        The number of subprocesses to use for data loading.
    dataset : torch.utils.data.Dataset
        The dataset to load data from.

    Returns
    -------
    float
        The total time taken to iterate through the dataset once.

    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Start profiling
    start_time = time.time()
    for _ in loader:
        pass  # Simulate training loop without computation
    end_time = time.time()
    return end_time - start_time


def run_profiling(config):
    """
    Runs profiling for data loading with various batch sizes and numbers of workers as specified in the config.

    Parameters
    ----------
    config : ProfilingConfig
        The configuration for profiling.

    Returns
    -------
    dict
        A dictionary mapping (batch_size, num_workers) to the total average time taken to iterate through the dataset once.
    """
    dataset = RotatedImageDataset(
        annotations_file=config.annotations_file, img_dir=config.img_dir, subset=config.subset
    )
    results = {}
    for batch_size in config.batch_sizes:
        for num_workers in config.num_workers:
            total_time = 0
            for _ in range(config.num_epochs):
                total_time += profile_data_loading(batch_size, num_workers, dataset)

            total_time /= config.num_epochs
            results[(batch_size, num_workers)] = total_time
    
    return results


def main():
    config = ProfilingConfig(
        img_dir="/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data",
        annotations_file="/scratch/gpfs/eh0560/datasets/deskewing/synthetic_image_angles.csv",
        subset="val",  # or "train" or "test"
        batch_sizes=[16, 32, 48, 64, 96],
        num_workers=[16, 20],
        num_epochs=3,
    )

    results = run_profiling(config)

    for params, time_taken in results.items():
        print(f"Batch Size: {params[0]}, Num Workers: {params[1]} -> Total Time: {time_taken:.2f} seconds")

    # Plot results
    batch_sizes = [params[0] for params in results.keys()]
    num_workers = [params[1] for params in results.keys()]
    times = list(results.values())

    fig, ax = plt.subplots()
    ax.scatter(batch_sizes, num_workers, c=times, cmap="viridis")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Number of Workers")
    ax.set_title("Data Loading Time (seconds)")
    fig.savefig("logs/data_loading_time.png")



if __name__ == "__main__":
    main()
