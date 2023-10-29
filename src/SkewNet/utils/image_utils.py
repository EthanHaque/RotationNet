import logging
import os
import cv2
import numpy as np
import math


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
        The angle to rotate the image by in radians.

    Returns
    -------
    numpy.ndarray
        The rotated image.
    """
    logger = logging.getLogger(__name__)
    height, width = image.shape[:2]
    new_width, new_height = get_size_of_rotated_image(width, height, angle)
    degrees = angle * 180 / math.pi

    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    logger.info(f"Rotated image by {angle} radians")
    return rotated_image


def superimpose_image_on_background(image, background, mask, x, y):
    """
    Superimpose an image on a background image at the specified coordinates.

    Parameters
    ----------
    image : numpy.ndarray
        The image to superimpose.
    background : numpy.ndarray
        The background image.
    mask : numpy.ndarray
        The mask for the image with the same shape as the image.
    x : int
        The x coordinate of the top left corner of the image.
    y : int
        The y coordinate of the top left corner of the image.

    Returns
    -------
    numpy.ndarray
        The superimposed image.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Superimposing image at ({x}, {y})")
    background_height, background_width = background.shape[:2]
    image_height, image_width = image.shape[:2]

    if x + image_width > background_width or y + image_height > background_height:
        logger.error("Image cannot be superimposed at specified coordinates")
        return background

    if mask is not None:
        region_of_interest = background[y:y + image_height, x:x + image_width]
        image_and_mask = cv2.bitwise_and(image, mask)
        not_mask = cv2.bitwise_not(mask)
        roi_and_not_mask = cv2.bitwise_and(region_of_interest, not_mask)
        blended_image = image_and_mask + roi_and_not_mask
        # blended_image = cv2.bitwise_and(image, mask) + cv2.bitwise_and(region_of_interest, cv2.bitwise_not(mask))
        background[y:y + image_height, x:x + image_width] = blended_image
    else:
        background[y:y + image_height, x:x + image_width] = image

    return background

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


def invert_image_channels(image):
    """
    Invert the channels of an image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to invert.

    Returns
    -------
    numpy.ndarray
        The inverted image.
    """
    logger = logging.getLogger(__name__)
    logger.info("Inverting image channels")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_size_of_rotated_image(width, height, angle):
    """Returns the new width and height of the image after rotation.

    Parameters
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.
    angle : float
        The angle to rotate the image by in radians.

    Returns
    -------
    tuple of int
        The new width and height of the image.
    """
    new_width = abs(width * math.cos(angle)) + abs(height * math.sin(angle))
    new_height = abs(width * math.sin(angle)) + abs(height * math.cos(angle))

    return int(new_width), int(new_height)


def get_largest_inscribed_rectangle_dimensions(width, height, angle):
    """Returns the width and height of the largest inscribed rectangle after rotation
    with sides parallel to the x and y axes.

    Parameters
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.
    angle : float
        The angle to rotate the image by in radians.

    Returns
    -------
    tuple of int
        The width and height of the largest inscribed rectangle.
    """
    #TODO: clean this method up.
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    longer_side, shorter_side = (width, height) if width_is_longer else (height, width)

    abs_sin_angle = abs(math.sin(angle))
    abs_cos_angle = abs(math.cos(angle))

    # Determine if the rectangle is half or fully constrained, and calculate dimensions
    if shorter_side <= 2 * abs_sin_angle * abs_cos_angle * longer_side or  abs(abs_sin_angle - abs_cos_angle) < 1e-10:
        # Half constrained case
        x = 0.5 * shorter_side
        inscribed_width, inscribed_height = (x / abs_sin_angle, x / abs_cos_angle) if width_is_longer else (x / abs_cos_angle, x / abs_sin_angle)
    else:
        # Fully constrained case
        cos_2angle = abs_cos_angle**2 - abs_sin_angle**2
        inscribed_width = (width * abs_cos_angle - height * abs_sin_angle) / cos_2angle
        inscribed_height = (height * abs_cos_angle - width * abs_sin_angle) / cos_2angle

    return int(inscribed_width), int(inscribed_height)


def flip_image(image, axis):
    """Flip an image along the specified axis.

    Parameters
    ----------
    image : numpy.ndarray
        The image to flip.
    axis : int
        The axis to flip the image along. 0 is vertical, 1 is horizontal, and -1 is both.

    Returns
    -------
    numpy.ndarray
        The flipped image.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Flipping image along axis {axis}")
    return cv2.flip(image, axis)


def crop_image(image, x1, y1, width, height):
    """Crop an image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to crop.
    x1 : int
        The x coordinate of the top left corner of the crop.
    y1 : int
        The y coordinate of the top left corner of the crop.
    width : int
        The width of the crop.
    height : int
        The height of the crop.

    Returns
    -------
    numpy.ndarray
        The cropped image.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cropping image from ({x1}, {y1}) to ({x1 + width}, {y1 + height})")
    return image[y1:y1 + height, x1:x1 + width]


def crop_from_angle(rotated_image, old_width, old_height, angle):
    """Crop an image from the specified angle. The image will be cropped to the largest inscribed rectangle.

    Parameters
    ----------
    rotated_image : numpy.ndarray
        The image to crop.
    old_width : int 
        The width of the image before rotation.
    old_height : int
        The height of the image before rotation.
    angle : float
        The angle to rotate the image by in radians.

    Returns
    -------
    numpy.ndarray
        The cropped image.
    """
    wr, hr = get_largest_inscribed_rectangle_dimensions(old_width, old_height, angle)

    # Now, calculate the position where this rectangle should be cropped from the center of the rotated image
    center_x = rotated_image.shape[1] // 2
    center_y = rotated_image.shape[0] // 2
    
    x1 = max(0, center_x - int(wr // 2))
    y1 = max(0, center_y - int(hr // 2))
    x2 = min(rotated_image.shape[1], center_x + int(wr // 2))
    y2 = min(rotated_image.shape[0], center_y + int(hr // 2))

    return rotated_image[y1:y2, x1:x2]