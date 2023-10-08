import logging
from pathlib import Path
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
    Calculate the angle between two points.

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
    return degrees


def convert_cudl_jpeg_path_to_png_path(jpeg_path):
    """
    Convert a CUDL JPEG image path to a CUDL PNG image path.

    Parameters
    ----------
    jpeg_path : Path
        The path to the JPEG image.

    Returns
    -------
    Path
        The path to the PNG image.
    """
    return jpeg_path.parent.parent / "segmented_images" / jpeg_path.parent.name / (jpeg_path.stem + ".png")


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


def main():
    """
    Main function to rotate all images in the segmented_images directory.
    """
    logger = logging.getLogger(__name__)
    setup_logging("rotate_images", log_dir="logs")

    root_output_dir = Path("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotated_images")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    images_root = Path("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/segmented_images")

    annotations_file_path = Path("/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/rotation_angles_annotations/images_01_10_test/annotations.xml")

    image_angles = get_image_angles(annotations_file_path, parse_cvat_for_images_xml_strategy)

    print(image_angles)


if __name__ == '__main__':
    main()
