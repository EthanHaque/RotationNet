import glob
import os
import logging
import torch
import random
import time
import datetime
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions
from utils import coco_utils


def setup_logging():
    """
    Configure the logging for the application.

    Set up logging to write to 'logs/segmentation.log' and also print to console.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"logs/segmentation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


def load_sam_model(checkpoint, model_type, device):
    """
    Load the SAM model from a checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to the model checkpoint.
    model_type : str
        Type of SAM model to load.
    device : str
        Device to load the model on, e.g., 'cuda' or 'cpu'.

    Returns
    -------
    SamPredictor
        An instance of SAM model predictor.
    """
    logger = logging.getLogger(__name__)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    logger.info(f"Loaded SAM model {model_type} from {checkpoint}")
    return SamPredictor(sam)


def load_dino_model(config_path, checkpoint_path, device):
    """
    Load the DINO model from a checkpoint.

    Parameters
    ----------
    config_path : str
        Path to the model configuration.
    checkpoint_path : str
        Path to the model checkpoint.
    device : str
        Device to load the model on.

    Returns
    -------
    groundingdino.models.GroundingDINO.groundingdino.GroundingDINO
        An instance of the grounding DINO model.
    """
    logger = logging.getLogger(__name__)
    model = load_model(config_path, checkpoint_path, device)
    logger.info(f"Loaded DINO model from {checkpoint_path}")
    return model


def get_files_from_dir(directory, filetype):
    """
    Get a list of all file paths of the specified filetype from a directory.

    Parameters
    ----------
    directory : str
        Directory to search for files.

    filetype : str
        The file extension to search for (e.g. ".jpg", ".png", ".txt").

    Returns
    -------
    list
        A list of file paths.
    """
    logger = logging.getLogger(__name__)
    # Ensure filetype string starts with a dot
    filetype = f".{filetype.lstrip('.')}"
    file_paths = glob.glob(f"{directory}/**/*{filetype}", recursive=True)
    logger.info(f"Found {len(file_paths)} {filetype} files in {directory}")
    return list(file_paths)


def predict_bounding_boxes(dino_model, image, device, prompt, box_threshold=0.45, text_threshold=0.25):
    """
    Predict bounding boxes for the given image using the DINO model.

    Parameters
    ----------
    dino_model : torch.nn.Module
        The DINO model instance.
    image : Tensor
        Image tensor.
    device : str
        Device to perform predictions on.
    prompt : str
        Text prompt for predictions.
    box_threshold : float, optional
        Confidence threshold for boxes. Defaults to 0.45.
    text_threshold : float, optional
        Confidence threshold for text. Defaults to 0.25.

    Returns
    -------
    tuple
        (boxes, logits, phrases) predicted by the model.
    """
    logger = logging.getLogger(__name__)
    boxes, logits, phrases = predict(
        dino_model, image, prompt, box_threshold, text_threshold, device
    )
    logger.info(f"Predicted {len(boxes)} bounding boxes")
    return boxes, logits, phrases


def predict_masks(sam_predictor, image_source, boxes, device):
    """
    Predict segmentation masks for the given image using the SAM model.

    Parameters
    ----------
    sam_predictor : SamPredictor
        The SAM model predictor instance.
    image_source : numpy.ndarray
        The source image array.
    boxes : Tensor
        Bounding boxes tensor.
    device : str
        Device to perform predictions on.

    Returns
    -------
    numpy.ndarray
        The predicted mask array.
    """
    logger = logging.getLogger(__name__)
    sam_predictor.set_image(image_source)

    height, width, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(device)

    logger.info(f"Predicting masks for {len(transformed_boxes)} boxes")
    masks, iou_preds, logits = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    logger.info(f"Predicted {len(masks)} masks")

    flattened_mask = masks.sum(dim=1)
    masks = (flattened_mask == True).cpu().numpy()

    areas = [np.sum(mask) for mask in masks]
    largest_mask_idx = np.argmax(areas) if areas else 0
    mask = masks[largest_mask_idx]
    logger.info(f"Flattened masks to shape {mask.shape}")

    mask, _ = remove_small_regions(mask, 100 * 100, "holes")
    mask, _ = remove_small_regions(mask, 100 * 100, "islands")
    logger.info(f"Removed small regions from mask")

    return mask


def initialize_models(device):
    """
    Load and initialize both SAM and DINO models.

    Parameters
    ----------
    device : str
        Device to load the models on.

    Returns
    -------
    tuple
        (sam_predictor, dino_model) initialized models.
    """
    logger = logging.getLogger(__name__)
    # Paths are hard-coded for this usecase
    sam_checkpoint = "/scratch/gpfs/eh0560/segment-anything/sam_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_predictor = load_sam_model(sam_checkpoint, model_type, device)

    dino_model_path = "/scratch/gpfs/eh0560/GroundingDINO/models/groundingdino_swinb_cogcoor.pth"
    dino_config_path = "/scratch/gpfs/eh0560/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    dino_model = load_dino_model(dino_config_path, dino_model_path, device)

    logger.info(f"Initialized models")
    return sam_predictor, dino_model


def process_image(image_path, sam_predictor, dino_model, device, text_prompt, box_threshold, text_threshold):
    """
    Process an image to extract segmentation masks.

    Parameters
    ----------
    image_path : str
        Path to the image to process.
    sam_predictor : SamPredictor
        The SAM model predictor instance.
    dino_model : torch.nn.Module
        The DINO model instance.
    device : str
        Device to perform predictions on.
    text_prompt : str
        Text prompt for bounding box predictions.
    box_threshold : float
        Confidence threshold for boxes.
    text_threshold : float
        Confidence threshold for text.

    Returns
    -------
    dict
        The Run-Length Encoding (RLE) of the predicted mask.
    """
    logger = logging.getLogger(__name__)
    image_source, image = load_image(image_path)
    boxes, _, _ = predict_bounding_boxes(dino_model, image, device, text_prompt, box_threshold, text_threshold)
    mask = predict_masks(sam_predictor, image_source, boxes, device)
    rle = coco_utils.convert_mask_to_rle(mask)
    logger.info(f"Processed image {image_path}")
    return rle


def main():
    """
    Main execution function to initialize models, load images, and generate masks.
    """
    start_time = time.time()
    max_minutes = 59

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting segmentation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    sam_predictor, dino_model = initialize_models(device)

    # settings are hard-coded for usecase
    text_prompt = "scanned old brown document"
    box_threshold = 0.45
    text_threshold = 0.25

    random.seed(42)
    image_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/images"
    image_paths = get_files_from_dir(image_dir, "jpg")
    random.shuffle(image_paths)

    masks_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/document_masks"
    processed_images = get_files_from_dir(masks_dir, "json")

    images_to_process = list(set(image_paths) - set(processed_images))

    for image_path in images_to_process:
        elapsed_time = time.time() - start_time
        if elapsed_time > max_minutes * 60:  # convert to seconds
            logger.info(f"Reached the maximum allowed time of {max_minutes} minutes. Stopping...")
            break

        try:
            rle_mask = process_image(image_path, sam_predictor, dino_model, device, text_prompt, box_threshold,
                                     text_threshold)

            image_directory = os.path.dirname(image_path).split("/")[-1]
            filename = os.path.basename(image_path).replace(".jpg", ".json")
            output_dir = os.path.join(masks_dir, image_directory)
            os.makedirs(output_dir, exist_ok=True)
            coco_utils.save_mask_as_rle(rle_mask, output_dir, filename)
        except Exception as e:
            logger.error(f"Failed to process image {image_path} with error {e}")
            continue

    logger.info(f"Finished segmentation")


if __name__ == '__main__':
    main()
