import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from utils import coco_utils
from utils.logging_utils import setup_logging


def apply_mask_and_crop(image, mask):
    """
    Apply a binary mask to the image and crop the resulting image.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C) where C is 1 for grayscale and 3 for RGB.
    mask : np.ndarray
        Binary mask of shape (H, W).

    Returns
    -------
    np.ndarray
        Masked image with an additional alpha channel and cropped to the mask boundaries.

    Raises
    ------
    ValueError
        If the dimensions of the image and mask do not match.
    """

    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match")

    h, w = image.shape[:2]

    # Check if the image is grayscale or RGB
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
        output = np.zeros((h, w, 2), dtype=np.uint8)  # 2 channels, the last one is alpha
        output[:, :, 0] = image.squeeze()
    else:  # RGB
        output = np.zeros((h, w, 4), dtype=np.uint8)  # 4 channels, the last one is alpha
        output[:, :, :3] = image

    output[:, :, -1] = mask * 255

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_output = output[rmin:rmax + 1, cmin:cmax + 1, :]
    return cropped_output


def find_image_mask_pairs(images_root, masks_root):
    """
    Find pairs of images and their corresponding mask files.

    Parameters
    ----------
    images_root : str
        Root directory where image files are stored.
    masks_root : str
        Root directory where mask files (in JSON format) are stored.

    Returns
    -------
    list of tuple
        List of pairs where each pair contains the path to the image file and its corresponding mask file.
    """

    mask_paths = glob.glob(os.path.join(masks_root, "**", "*.json"), recursive=True)

    pairs = []
    for mask_path in mask_paths:
        relative_path = os.path.relpath(mask_path, masks_root)
        base, _ = os.path.splitext(relative_path)
        image_path = os.path.join(images_root, base + ".jpg")

        if os.path.exists(image_path):
            pairs.append((image_path, mask_path))

    return pairs


def save_masked_image(image_path, mask_path, output_directory, images_root):
    """
    Apply mask to an image, crop it, and save the result to the specified directory.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    mask_path : str
        Path to the mask file (in RLE format).
    output_directory : str
        Directory where the masked image will be saved.
    images_root : str
        Root directory where image files are stored.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing image: {image_path} with mask: {mask_path}")
    relative_path = os.path.relpath(image_path, images_root)
    folder_structure = os.path.dirname(relative_path)
    final_output_directory = os.path.join(output_directory, folder_structure)

    os.makedirs(final_output_directory, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = np.asarray(image)

    rle = coco_utils.load_rle_from_file(mask_path)
    mask = mask_utils.decode(rle)

    masked_image = apply_mask_and_crop(image, mask)

    base_name = os.path.basename(image_path)
    name_without_extension, _ = os.path.splitext(base_name)
    output_path = os.path.join(final_output_directory, f"{name_without_extension}.png")
    cv2.imwrite(output_path, masked_image)
    logger.info(f"Saved masked image to: {output_path}")


def process_image(pair, output_directory, images_root):
    """
    Helper function to process a single image-mask pair.
    """
    logger = logging.getLogger(__name__)
    image_path, mask_path = pair
    try:
        save_masked_image(image_path, mask_path, output_directory, images_root)
    except Exception as e:
        logger.error(f"Failed to process image {image_path} with error {e}")


def main():
    """
    Main function to apply masks to a set of images and save the results.
    """
    setup_logging("masking", log_dir="logs")
    logger = logging.getLogger(__name__)

    images_root = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/images"
    masks_root = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/document_masks"
    output_directory = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/segmented_images"

    pairs = find_image_mask_pairs(images_root, masks_root)

    logger.info(f"Starting mask processing with {cpu_count()} cores.")

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        logger.info(f"Using {cpu_count()} cores to process images.")
        list(executor.map(process_image, pairs, [output_directory] * len(pairs), [images_root] * len(pairs)))

    logger.info("Finished processing images.")


if __name__ == '__main__':
    main()
