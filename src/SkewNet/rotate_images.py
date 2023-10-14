import logging
from pathlib import Path
import cv2
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from utils.image_utils import get_files, rotate_image
from utils.logging_utils import setup_logging
import xml.etree.ElementTree as ET
import math


def parse_cvat_for_images_xml_strategy(cvat_xml_path):
    """
    Parse the CVAT XML file to get the image names and rotation angles.

    Parameters
    ----------
    cvat_xml_path : str
        The path to the CVAT XML file.

    Returns
    -------
    dict
        A dictionary mapping image names to rotation angles.
    """
    logger = logging.getLogger(__name__)
    image_angles = {}

    tree = ET.parse(cvat_xml_path)
    root = tree.getroot()

    for image in root.findall('.//image'):
        image_name = image.get('name')

        polyline = image.find('.//polyline')
        if polyline is not None:
            try:
                points = polyline.get('points').split(';')
                point1 = tuple(map(float, points[0].split(',')))
                point2 = tuple(map(float, points[1].split(',')))
                rotation_angle = calculate_angle(point1, point2)

                image_angles[image_name] = rotation_angle
            except Exception as e:
                logger.error(f"Could not calculate the angle for image {image_name}: {e}")

    return image_angles


def calculate_angle(point1, point2):
    """
    Calculate the angle between two points, with 0 degrees being parallel to the negative x-axis.

    Parameters
    ----------
    point1 : tuple
        The first point as (x, y).
    point2 : tuple
        The second point as (x, y).

    Returns
    -------
    float
        The angle in degrees.
    """
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    radians = math.atan2(delta_y, delta_x)
    degrees = math.degrees(radians)
    adjusted_degrees = (degrees + 180) % 360
    return adjusted_degrees


def get_image_angles(annotations_file_path, strategy):
    """
    Get the rotation angles for each image in the annotations file.

    Parameters
    ----------
    annotations_file_path : str
        The path to the annotations file.
    strategy : function
        The function to use to parse the annotations file.

    Returns
    -------
    dict
        A dictionary mapping image names to rotation angles.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Using strategy {strategy.__name__} to parse {annotations_file_path}")
    return strategy(annotations_file_path)


def rotate_and_save_image(segmented_image_path, angle, rotated_images_root):
    """
    Load, rotate and save the image in parallel.

    Parameters
    ----------
    segmented_image_path : Path
        The path to the segmented image.
    angle : float
        The rotation angle.
    rotated_images_root : Path
        The path where to save the rotated images.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Rotating {segmented_image_path} by {angle} degrees")
        image = cv2.imread(str(segmented_image_path), cv2.IMREAD_UNCHANGED)
        rotated_image = rotate_image(image, angle)

        # Save the rotated image to the rotated_images_root directory.
        # This is slightly different from the convention images_01, images_02, etc.
        # Working with a small subset of the images, so this is no longer necessary.
        output_path = rotated_images_root / segmented_image_path.name
        cv2.imwrite(str(output_path), rotated_image)
        logger.info(f"Saved rotated image to {output_path}")
    except Exception as e:
        logger.error(f"Error rotating image {segmented_image_path}: {e}")
        raise e


def main():
    """
    Main function to rotate all images in the segmented_images directory.
    """
    setup_logging("rotate_images", log_dir="logs")
    logger = logging.getLogger(__name__)

    base_path = Path("/scratch/gpfs/RUSTOW/")
    annotations_file_path = base_path / "deskewing_datasets/images/cudl_images/rotation_angles_annotations/images_01_10_test/annotations.xml"
    rotated_images_root = base_path / "deskewing_datasets/images/cudl_images/rotated_images"
    rotated_images_root.mkdir(parents=True, exist_ok=True)

    image_angles = get_image_angles(str(annotations_file_path), parse_cvat_for_images_xml_strategy)
    # transforming the image_angles keys into absolute paths
    image_angles = {base_path / image_path: angle for image_path, angle in image_angles.items()}

    logger.info(f"Found {len(image_angles)} images")

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = []
        for jpeg_image_path, angle in image_angles.items():
            # small hack to get the segmented image path from the jpeg image path
            segmented_image_path = Path(str(jpeg_image_path).replace("/jpeg_images/", "/segmented_images/")).with_suffix(".png")
            futures.append(executor.submit(
                rotate_and_save_image,
                segmented_image_path,
                angle,
                rotated_images_root
            ))

        for future in futures:
            future.result() 

    logger.info("Finished rotating images")


if __name__ == '__main__':
    main()
