import uuid

import cv2
import flip
import numpy as np


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

    document_image_width = document_image.shape[1]
    document_image_height = document_image.shape[0]

    background_blur_strength = np.random.uniform(0.0, 1.0)

    transform_backgrounds = [
        flip.transformers.data_augmentation.Rotate(mode="random", force=False, crop=True),
        flip.transformers.data_augmentation.Flip("random", force=False),
        flip.transformers.data_augmentation.RandomResize(
            mode="symmetric_w",
            w_max=document_image_width * 2,
            h_max=document_image_height * 2,
            w_min=document_image_width,
            h_min=document_image_height,
            force=True,
        ),
        flip.transformers.data_augmentation.Noise("gaussian_blur", value=background_blur_strength, force=False),
    ]

    # Transformations to apply to the document image i.e. children of the background image
    transform_objects = [
        flip.transformers.data_augmentation.RandomResize(
            mode="symmetric_w",
            relation="parent",
            w_percentage_min=0.3,
            w_percentage_max=1.0,
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


def main():
    """
    Main function to test the synthetic data generation.
    """
    for i in range(10):
        document_image = (
            "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotated_images/MS-ADD-01611-000-00063.png"
        )
        background_image = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/texture_ninja/wood-8683.jpg"

        document_image = cv2.imread(document_image, cv2.IMREAD_UNCHANGED)
        background_image = cv2.imread(background_image, cv2.IMREAD_UNCHANGED)

        output_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"

        compose_document_onto_background(document_image, background_image, output_dir)


if __name__ == "__main__":
    main()
