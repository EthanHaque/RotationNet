import uuid

import cv2
import flip
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import time
from utils import logging_utils
import logging


def compose_document_onto_background(document_image, background_image, output_images_dir, output_annotations_dir):
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
    """
    logger = logging.getLogger(__name__)
    document_image = flip.utils.inv_channels(document_image)
    background_image = flip.utils.inv_channels(background_image)

    document_element = flip.transformers.Element(image=document_image, name="document")
    background_element = flip.transformers.Element(
        image=background_image, name="background", objects=[document_element]
    )

    document_angle = np.random.uniform(0.0, 360.0)
    background_angle = np.random.uniform(0.0, 360.0)
    # with probability 1/2 set background angle to 0
    if np.random.randint(0, 2) == 0:
        background_angle = 0.0

    rotated_document_width, rotated_document_height = flip.utils.get_size_of_rotated_bounding_box(
        document_image.shape[1], document_image.shape[0], document_angle
    )
    rotated_background_width, rotated_background_height = flip.utils.get_size_of_rotated_bounding_box(
        background_image.shape[1], background_image.shape[0], background_angle
    )
    rotated_background_width, rotated_background_height = flip.utils.largest_inscribed_rectangle(
        rotated_background_width, rotated_background_height, background_angle
    )

    largest_document_dimension = max(rotated_document_width, rotated_document_height)
    largest_background_dimension = max(rotated_background_width, rotated_background_height)
    smallest_dimension = min(largest_document_dimension, largest_background_dimension)
    # print all the dimensions
    # print(f"rotated_document_width: {rotated_document_width:.2f}, rotated_document_height: {rotated_document_height:.2f}, rotated_background_width: {rotated_background_width:.2f}, rotated_background_height: {rotated_background_height:.2f}, largest_document_dimension: {largest_document_dimension:.2f}, largest_background_dimension: {largest_background_dimension:.2f}, smallest_dimension: {smallest_dimension:.2f}")
    background_blur_strength = np.random.uniform(0.0, 1.0)

    transform_backgrounds = [
        flip.transformers.data_augmentation.Rotate(mode="by_angle", angle=background_angle, force=True, crop=True),
        flip.transformers.data_augmentation.Flip("random", force=False),
        flip.transformers.data_augmentation.RandomResize(
            mode="asymmetric",
            w_max=smallest_dimension * 1.7,
            h_max=smallest_dimension * 1.7,
            w_min=smallest_dimension * 1.0,
            h_min=smallest_dimension * 1.0,
            force=True,
        ),
        flip.transformers.data_augmentation.Noise("gaussian_blur", value=background_blur_strength, force=False),
    ]

    # Transformations to apply to the document image i.e. children of the background image
    transform_objects = [
        flip.transformers.data_augmentation.Rotate(mode="by_angle", angle=document_angle, force=True, crop=False),
        flip.transformers.data_augmentation.RandomResize(
            mode="larger",
            w_min=smallest_dimension * 0.8,
            w_max=smallest_dimension * 0.95,
            h_min=smallest_dimension * 0.8,
            h_max=smallest_dimension * 0.95,
            force=True,
        ),
    ]

    name = uuid.uuid4()
    transform = flip.transformers.Compose(
        [
            flip.transformers.ApplyToBackground(transform_backgrounds),
            flip.transformers.ApplyToObjects(transform_objects),
            flip.transformers.domain_randomization.ObjectsRandomPosition(
                x_min=0, y_min=0, x_max=1, y_max=1, mode="percentage"
            ),
            flip.transformers.domain_randomization.Draw(),
            flip.transformers.labeler.CreateBoundingBoxes(),
            flip.transformers.labeler.CreateAngles(),
            flip.transformers.io.SaveImage(output_images_dir, name),
            flip.transformers.io.CreateJson(output_annotations_dir, name),
        ]
    )

    [background_element] = transform(background_element)

    logger.info(f"Saved image {name} with {len(background_element.objects)} objects")

    return background_element


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


def process_image(image_path, background_path, output_images_dir, output_annotations_dir, index):
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

    output_annotations_dir : str
        Output directory where the annotations will be saved.

    index : int
        Index of the image to be saved.
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

    compose_document_onto_background(document_image, background_image, output_images_dir, output_annotations_dir)


def main():
    """
    Main function to test the synthetic data generation.
    """
    logging_utils.setup_logging("synthetic_data_generation", log_level=logging_utils.logging.INFO)
    logger = logging.getLogger(__name__)

    cudl_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotated_images")
    doc_lay_net_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/doc_lay_net/images")
    publaynet_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/publaynet/train")

    texture_ninja_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/texture_ninja")
    pexels_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/pexels_textures")
    # TODO: add images with solid colors as backgrounds

    output_images_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"
    output_annotations_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data_annotations"

    document_images = []

    # for i in range(10):
    #     document_images.extend(cudl_document_images)
    # document_images.extend(doc_lay_net_document_images)
    document_images.extend(publaynet_document_images)

    background_images = []
    background_images.extend(texture_ninja_background_images)
    background_images.extend(pexels_background_images)

    logger.info(f"Found {len(document_images)} document images")
    logger.info(f"Found {len(background_images)} background images")

    random_background_images = np.random.choice(background_images, len(document_images))

    with Pool(cpu_count()) as p:
        p.starmap(
            process_image,
            zip(
                document_images,
                random_background_images,
                [output_images_dir] * len(document_images),
                [output_annotations_dir] * len(document_images),
                range(len(document_images)),
            ),
        )

    # for i in range(len(document_images)):
    #     process_image(document_images[i], random_background_images[i], output_dir, i)

if __name__ == "__main__":
    main()
