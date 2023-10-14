import logging

import cv2
import numpy as np


def convert_png_to_jpeg(png_path, output_directory, new_name=None,
                        background_color=(255, 255, 255)):
    """
    Convert a PNG image to JPEG format and save it to the specified directory. If no new name is specified, the
    original name will be used. The new name should not include the file extension.

    Parameters
    ----------
    png_path : Path
        Path to the PNG image.
    output_directory : Path
        Directory where the JPEG image will be saved.
    new_name : str, optional
        New name for the JPEG image (without the file extension).
    background_color : tuple, optional
        RGB color to fill the alpha channel, default is white (255, 255, 255).
    """
    logger = logging.getLogger(__name__)
    png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)

    if png is None:
        logger.error(f"Could not read image {png_path}")
        return

    if png.shape[2] == 4:
        alpha_channel = png[:, :, 3]
        rgb_channels = png[:, :, :3]

        background = np.full_like(rgb_channels, background_color)
        final_rgb = (rgb_channels * (alpha_channel / 255.0)[:, :, None] +
                     background * (1 - (alpha_channel / 255.0)[:, :, None]))
        final_rgb = final_rgb.astype(np.uint8)
    else:
        final_rgb = png[:, :, :3]

    if new_name is None:
        new_name = png_path.stem + ".jpg"
    else:
        new_name = new_name + ".jpg"

    logger.info(f"Saving {new_name} to {output_directory}")
    jpeg_path = output_directory / new_name
    cv2.imwrite(str(jpeg_path), final_rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
    logger.info(f"Saved {new_name} to {output_directory}")


def get_files(input_folder, extension):
    """
    Get all file in the specified directory with the specified extension.

    Parameters
    ----------
    input_folder : Path
        The directory to search for PNG images.
    extension : str
        The file extension to search for.

    Returns
    -------
    list of Path
    """
    logger = logging.getLogger(__name__)
    if extension[0] != ".":
        extension = "." + extension
    files = list(input_folder.glob(f"*{extension}"))
    logger.info(f"Found {len(files)} files with extension {extension} in {input_folder}")
    return files


def rotate_image(image, angle):
    """
    Rotate an image by the specified angle.

    Parameters
    ----------
    image : numpy.ndarray
        The image to rotate.
    angle : float
        The angle to rotate the image by in degrees.

    Returns
    -------
    numpy.ndarray
        The rotated image.
    """
    logger = logging.getLogger(__name__)
    height, width = image.shape[:2]

    # Handle the case for images with alpha channel
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)

        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        b = cv2.warpAffine(b, rotation_matrix, (width, height))
        g = cv2.warpAffine(g, rotation_matrix, (width, height))
        r = cv2.warpAffine(r, rotation_matrix, (width, height))
        a = cv2.warpAffine(a, rotation_matrix, (width, height))

        rotated_image = cv2.merge((b, g, r, a))
    else:
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    logger.info(f"Rotated image by {angle} degrees")
    return rotated_image
