import logging
import os
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


def convert_bytes_to_jpeg(file_contents, output_path):
    """
    Convert the file contents to a JPEG image and save it to the specified path.

    Parameters
    ----------
    file_contents : bytes
        The contents of the image file.
    output_path : str
        The path to save the converted image to.
    """
    array = np.frombuffer(file_contents, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def convert_bytes_to_jpeg_and_resize(file_contents, output_path, largest_dimension):
    """
    Convert the file contents to a JPEG image, resize it to the specified largest dimension, and save it to the
    specified path.

    Parameters
    ----------
    file_contents : bytes
        The contents of the image file.
    output_path : str
        The path to save the converted image to.
    largest_dimension : int
        The largest dimension of the resized image.
    """
    if file_contents is None:
        # temp 
        print("output_path: ", output_path)
        raise ValueError("File contents cannot be None")
    array = np.frombuffer(file_contents, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    if height > width:
        new_height = largest_dimension
        new_width = int(width * (largest_dimension / height))
    else:
        new_width = largest_dimension
        new_height = int(height * (largest_dimension / width))

    img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


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


def get_file_name_without_extension(file):
    """
    Get the name of the file without the extension.

    Parameters
    ----------
    file : Path
        The path to the file.

    Returns
    -------
    str
        The name of the file without the extension.
    """
    name, _ = os.path.splitext(os.path.basename(file))
    return name