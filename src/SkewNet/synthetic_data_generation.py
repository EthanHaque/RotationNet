import uuid

import cv2
import flip
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import time


def compose_document_onto_background(document_image, background_image, output_dir):
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
    document_image = flip.utils.inv_channels(document_image)
    background_image = flip.utils.inv_channels(background_image)

    document_element = flip.transformers.Element(image=document_image, name="document")
    background_element = flip.transformers.Element(
        image=background_image, name="background", objects=[document_element]
    )
    
    largest_document_dimension = max(document_image.shape[0], document_image.shape[1])
    largest_background_dimension = max(background_image.shape[0], background_image.shape[1])
    smallest_dimension = min(largest_document_dimension, largest_background_dimension)

    background_blur_strength = np.random.uniform(0.0, 1.0)

    transform_backgrounds = [
        flip.transformers.data_augmentation.Rotate(mode="random", force=False, crop=True),
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
        flip.transformers.data_augmentation.RandomResize(
            mode="larger",
            w_min = smallest_dimension * 0.8,
            w_max = smallest_dimension * 1.0,
            h_min = smallest_dimension * 0.8,
            h_max = smallest_dimension * 1.0,
            force=True,
        ),
        flip.transformers.data_augmentation.Rotate(mode="random", force=True, crop=False),
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
            flip.transformers.io.SaveImage(output_dir, name),
        ]
    )

    [background_element] = transform(background_element)

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


def process_image(image_path, background_path, output_dir, index):
    """
    Composes a document image onto a background image applying a series
    of transformations to the document image and the background image.

    Parameters
    ----------
    image_path : str
        Path to the document image.

    background_path : str
        Path to the background image.

    output_dir : str
        Output directory where the composed image will be saved.

    index : int
        Index of the image to be saved.
    """
    np.random.seed(int(time.time()) + index)

    document_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    background_image = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)

    compose_document_onto_background(document_image, background_image, output_dir)



def main():
    """
    Main function to test the synthetic data generation.
    """
    # with Pool(cpu_count()) as p:
    #     p.map(test_process_image, range(100))

    cudl_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotated_images")
    doc_lay_net_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/doc_lay_net/images")
    publaynet_document_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/publaynet/train")

    texture_ninja_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/texture_ninja")
    pexels_background_images = collect_files("/scratch/gpfs/RUSTOW/deskewing_datasets/images/pexels_textures")

    output_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"


    document_images = []

    for i in range(10):
        document_images.extend(cudl_document_images)
    # document_images.extend(doc_lay_net_document_images)
    # document_images.extend(publaynet_document_images)

    background_images = []
    background_images.extend(texture_ninja_background_images)
    background_images.extend(pexels_background_images)

    random_background_images = np.random.choice(background_images, len(document_images))

    # multiprocess the image generation
    with Pool(cpu_count()) as p:
        p.starmap(process_image, zip(document_images, random_background_images, [output_dir] * len(document_images), range(len(document_images))))


if __name__ == "__main__":
    main()


