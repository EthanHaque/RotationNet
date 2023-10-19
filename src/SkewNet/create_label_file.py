import json
import os
import glob
import pandas as pd
from utils.logging_utils import setup_logging
import logging


def get_json_files(directory):
    """
    Get a list of all JSON files in a given directory.

    Parameters
    ----------
    directory : str
        The directory containing JSON files.

    Returns
    -------
    list
        A list containing the paths of all JSON files in the specified directory.
    """
    return glob.glob(os.path.join(directory, "*.json"))


def extract_angle_from_json(json_file):
    """
    Extract angle information from a given JSON file.

    Parameters
    ----------
    json_file : str
        The path to the JSON file.

    Returns
    -------
    float
        The extracted angle value.

    Raises
    ------
    ValueError
        If the JSON file has an invalid format or the angle is not found.
    json.JSONDecodeError
        If the file is not a valid JSON.
    """
    with open(json_file, 'r') as jfile:
        data = json.load(jfile)

        # this might change in the future, kida hacky but it works for now
        angle = data[1].get('angle')
        if angle is None:
            raise ValueError(f"Angle not found in {json_file}")

    return angle


def create_angle_dataframe(json_directory, image_directory):
    """
    Create a DataFrame containing filenames and their corresponding angles.

    Parameters
    ----------
    json_directory : str
        The directory containing JSON files.
    image_directory : str
        The directory containing image files.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing filenames and angles.

    Raises
    ------
    FileNotFoundError
        If the corresponding image file for a JSON file is not found.
    """
    json_files = get_json_files(json_directory)
    data = []

    for json_file in json_files:
        image_filename = os.path.basename(json_file).rsplit('.', 1)[0] + ".jpg"
        image_path = os.path.join(image_directory, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        try:
            angle = extract_angle_from_json(json_file)
            data.append((image_filename, angle))
        except (ValueError, json.JSONDecodeError) as e:
            print(e)
            continue

    return pd.DataFrame(data, columns=['filename', 'angle'])


def main(json_directory, image_directory, output_csv):
    """
    Main function to create a CSV file mapping image filenames to angles.

    Parameters
    ----------
    json_directory : str
        Path to the directory containing JSON files with angle information.
    image_directory : str
        Path to the directory containing image files.
    output_csv : str
        Path to the output CSV file to be created.

    Returns
    -------
    None
    """
    # TODO: add train, test, val split
    setup_logging("create_label_file", log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    df = create_angle_dataframe(json_directory, image_directory)
    df.to_csv(output_csv, index=False)
    logger.info(f"CSV file saved to {output_csv}")


if __name__ == "__main__":
    json_directory = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data_annotations"
    image_directory = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_data"
    output_csv = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/synthetic_image_angles.csv"

    main(json_directory, image_directory, output_csv)
