import base64
import json
import os

import numpy as np
from pycocotools import mask as coco_mask


def convert_mask_to_rle(mask):
    """
    Convert a binary mask into its Run Length Encoding (RLE) representation.

    Parameters
    ----------
    mask : numpy.ndarray
        A 2D binary mask to be encoded.

    Returns
    -------
    dict
        The RLE representation of the input mask.
    """
    rle = coco_mask.encode(np.asfortranarray(mask))
    return rle


def rle_to_serializable(rle):
    """
    Convert RLE data into a serializable format suitable for JSON encoding.

    Parameters
    ----------
    rle : dict
        The RLE data with 'counts' as bytes.

    Returns
    -------
    dict
        The serializable RLE representation with 'counts' as a base64 encoded string.
    """
    serializable_rle = rle.copy()
    # Convert bytes to base64 encoded string
    serializable_rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
    return serializable_rle


def serializable_to_rle(serializable_rle):
    """
    Convert a serializable RLE format back to its original RLE format.

    Parameters
    ----------
    serializable_rle : dict
        The serializable RLE data with 'counts' as a base64 encoded string.

    Returns
    -------
    dict
        The original RLE representation with 'counts' as bytes.
    """
    rle = serializable_rle.copy()
    # Convert base64 encoded string to bytes
    rle['counts'] = base64.b64decode(serializable_rle['counts'].encode('utf-8'))
    return rle


def save_mask_as_rle(rle, output_dir, filename):
    """
    Save an RLE mask representation to a file in JSON format.

    Parameters
    ----------
    rle : dict
        The RLE data to save.
    output_dir : str
        The directory path where the file should be saved.
    filename : str
        The name of the output file.

    Returns
    -------
    None
    """
    serializable_rle = rle_to_serializable(rle)
    serialized_json = json.dumps(serializable_rle)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(serialized_json)


def load_rle_from_file(filename):
    """
    Load RLE mask data from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the input file containing the RLE data in JSON format.

    Returns
    -------
    dict
        The loaded RLE mask data.
    """
    with open(filename, 'r') as f:
        serializable_rle = json.load(f)
    return serializable_to_rle(serializable_rle)
