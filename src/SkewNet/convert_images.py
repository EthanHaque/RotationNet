import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from utils.image_utils import get_files, convert_png_to_jpeg
from utils.logging_utils import setup_logging


def process_task(task):
    """
    Helper function to process a single image.

    Parameters
    ----------
    task : tuple
        A tuple containing the image path and the output directory.
    """
    img, current_output_dir = task
    convert_png_to_jpeg(img, current_output_dir)


def main():
    """
    Main function to convert all images in the segmented_images directory to JPEG images.
    """
    setup_logging("convert_images", log_dir="logs")
    logger = logging.getLogger(__name__)
    root_output_dir = Path("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/jpeg_images")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    images_root = Path("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/segmented_images")
    subdirectories = [x for x in images_root.iterdir() if x.is_dir()]

    tasks = []
    for subdirectory in subdirectories:
        logger.info(f"Processing {subdirectory}")
        png_images = get_files(subdirectory)
        logger.info(f"Found {len(png_images)} images")

        current_output_dir = root_output_dir / subdirectory.name
        current_output_dir.mkdir(parents=True, exist_ok=True)

        tasks.extend((img, current_output_dir) for img in png_images)

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        list(executor.map(process_task, tasks))

    logger.info("Finished converting images")


if __name__ == "__main__":
    main()
