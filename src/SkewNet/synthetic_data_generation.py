import logging
import os
import time
import uuid
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from utils import image_utils, logging_utils
import pandas as pd
from sklearn.model_selection import train_test_split


def compose_document_onto_background(document_image, background_image, output_images_dir):
    """
    Composes a document image onto a background image applying a series
    of transformations to the document image and the background image.

    Parameters
    ----------
    document_image : numpy.ndarray
        Document image to be composed onto the background image.
    background_image : numpy.ndarray
        Background image.
    output_dir : str
        Output directory where the composed image will be saved.

    Returns
    -------
    annotation : dict
        Dictionary containing the annotation information.
    """
    logger = logging.getLogger(__name__)

    # randomly scaling the document image and changing aspect ratio
    random_x_scale = np.random.uniform(0.75, 1.1)
    random_y_scale = np.random.uniform(0.75, 1.1)
    document_image = cv2.resize(document_image, (0, 0), fx=random_x_scale, fy=random_y_scale)

    initial_backgound_height, initial_backgound_width = background_image.shape[:2]
    initial_document_height, initial_document_width = document_image.shape[:2]

    document_angle = np.random.uniform(0.0, 30.0) * np.pi / 180.0
    background_angle = np.random.uniform(0.0, 360.0) * np.pi / 180.0
    target_background_width = 900
    target_background_height = 1200

    rotated_document_width, rotated_document_height = image_utils.get_size_of_rotated_image(document_image.shape[1], document_image.shape[0], document_angle)
    rotated_background_width, rotated_background_height = image_utils.get_size_of_rotated_image(background_image.shape[1], background_image.shape[0], background_angle)

    scale_down_factor = np.random.uniform(0.75, 0.95)
    largest_document_side = max(rotated_document_width, rotated_document_height)
    smallest_target_size = min(target_background_width, target_background_height)
    scale = scale_down_factor * smallest_target_size / largest_document_side

    document_image = cv2.resize(document_image, (0, 0), fx=scale, fy=scale)
    mask = np.ones(document_image.shape, dtype=np.uint8) * 255

    document_image = image_utils.rotate_image(document_image, document_angle)
    mask = image_utils.rotate_image(mask, document_angle)

    random_flip_direciton = np.random.randint(-1, 2)
    background_image = image_utils.flip_image(background_image, random_flip_direciton)
    background_image = image_utils.rotate_image(background_image, background_angle)
    background_image = image_utils.crop_from_angle(background_image, initial_backgound_width, initial_backgound_height, -background_angle)

    random_width_scale = np.random.uniform(1.0, 1.1)
    random_height_scale = np.random.uniform(1.0, 1.1)

    # subtract 1 to avoid out of bounds error
    crop_width = int(target_background_width * random_width_scale)
    crop_width = min(crop_width, background_image.shape[1]) - 1
    crop_height = int(target_background_height * random_height_scale) 
    crop_height = min(crop_height, background_image.shape[0]) - 1

    random_crop_x1 = np.random.randint(0, background_image.shape[1] - crop_width)
    random_crop_y1 = np.random.randint(0, background_image.shape[0] - crop_height)


    background_image = image_utils.crop_image(background_image, random_crop_x1, random_crop_y1, crop_width, crop_height)
    background_image = cv2.resize(background_image, (target_background_width, target_background_height))

    superimposed_image_x = np.random.randint(0, background_image.shape[1] - document_image.shape[1])
    superimposed_image_y = np.random.randint(0, background_image.shape[0] - document_image.shape[0])

    superimposed_image = image_utils.superimpose_image_on_background(document_image, background_image, mask, superimposed_image_x, superimposed_image_y)

    name = uuid.uuid4()
    cv2.imwrite(os.path.join(output_images_dir, f"{name}.jpg"), superimposed_image)

    annotation = {
        "image_name": f"{name}.jpg",
        "document_angle": document_angle,
        "background_angle": background_angle,
        "scale": scale,
        "random_flip_direciton": random_flip_direciton,
        "random_width_scale": random_width_scale,
        "random_height_scale": random_height_scale,
        "random_crop_x1": random_crop_x1,
        "random_crop_y1": random_crop_y1,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "superimposed_image_x": superimposed_image_x,
        "superimposed_image_y": superimposed_image_y,

    }

    return annotation


def collect_files(directory):
    """
    Collects all files in a directory

    Parameters
    ----------
    directory : str
        Directory to search for files.

    Returns
    -------
    files : list
        List of files.
    """
    if not os.path.isdir(directory):
        raise ValueError("Directory does not exist")

    return [os.path.join(directory, f) for f in os.listdir(directory)]


def process_image(image_path, background_path, output_images_dir, index):
    """
    Composes a document image onto a background image applying a series
    of transformations to the document image and the background image.

    Parameters
    ----------
    image_path : str
        Path to the document image.
    background_path : str
        Path to the background image.
    output_images_dir : str
        Output directory where the composed image will be saved.
    index : int
        Index of the image to be saved.

    Returns
    -------
    annotation : dict
        Dictionary containing the annotation information.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing image {index} with path {image_path}")

    np.random.seed(int(time.time()) + index)

    document_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if (document_image.shape[2]) == 3:
        document_image = cv2.cvtColor(document_image, cv2.COLOR_BGR2BGRA)

    background_image = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    if (background_image.shape[2]) == 3:
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)

    
    if background_image.dtype != np.uint8:
        logger.warning(f"Background image {background_path} has dtype {background_image.dtype}")
        return

    annotation = compose_document_onto_background(document_image, background_image, output_images_dir)
    return annotation


def main():
    """
    Main function to test the synthetic data generation.
    """
    logging_utils.setup_logging("synthetic_data_generation", log_level=logging_utils.logging.INFO, log_to_stdout=False) 
    logger = logging.getLogger(__name__)

    annotations_file = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_image_angles.csv"

    cudl_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotated_images")
    doc_lay_net_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/doc_lay_net/images")
    publaynet_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/publaynet/train")

    texture_ninja_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/texture_ninja")
    pexels_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/pexels_textures")
    # TODO: add images with solid colors as backgrounds and simple textures

    output_images_dir = "/scratch/gpfs/eh0560/datasets/deskewing/synthetic_data"

    os.makedirs(output_images_dir, exist_ok=True)

    document_images = []

    # for i in range(10):
    #     document_images.extend(cudl_document_images)
    document_images.extend(doc_lay_net_document_images)
    document_images.extend(publaynet_document_images)
    # document_images = [image for image in document_images for _ in range(5)]

    background_images = []
    background_images.extend(texture_ninja_background_images)
    background_images.extend(pexels_background_images)

    logger.info(f"Found {len(document_images)} document images")
    logger.info(f"Found {len(background_images)} background images")

    random_background_images = np.random.choice(background_images, len(document_images))

    annotations = []
    workers = cpu_count()
    logger.info(f"Using {workers} workers to process images")
    with Pool(workers) as p:
        annotations = p.starmap(
            process_image,
            zip(
                document_images,
                random_background_images,
                [output_images_dir] * len(document_images),
                range(len(document_images)),
            ),
        )

    annotations = [annotation for annotation in annotations if annotation is not None]
    annotations_df = pd.DataFrame(annotations)
    # train_test_spit(annotations_df, test_size=0.3, train_size=0.7)
    train_df, test_df = train_test_split(annotations_df, test_size=0.3, train_size=0.7)
    val_df, test_df = train_test_split(test_df, test_size=2/3, train_size=1/3)

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    final_df = pd.concat([train_df, test_df, val_df])

    final_df.to_csv(annotations_file, index=False)
    logger.info(f"Saved annotations to {annotations_file}")

    # for i in range(len(document_images)):
    #     process_image(document_images[i], random_background_images[i], output_images_dir, i)


if __name__ == "__main__":
    main()
