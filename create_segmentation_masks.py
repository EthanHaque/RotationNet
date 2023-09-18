import glob
import json
import os
import base64
import logging
import torch
import numpy as np
from pycocotools import mask as coco_mask
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("segmentation.log"), logging.StreamHandler()]
    )


def load_sam_model(checkpoint, model_type, device):
    logger = logging.getLogger(__name__)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    logger.info(f"Loaded SAM model {model_type} from {checkpoint}")
    return SamPredictor(sam)


def load_dino_model(config_path, checkpoint_path, device):
    logger = logging.getLogger(__name__)
    model = load_model(config_path, checkpoint_path, device)
    logger.info(f"Loaded DINO model from {checkpoint_path}")
    return model


def get_images_from_dir(image_dir):
    logger = logging.getLogger(__name__)
    image_paths = glob.glob(image_dir + "/*/*.jpg")
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    return image_paths


def predict_bounding_boxes(
        dino_model, image, device, prompt, box_threshold=0.45, text_threshold=0.25
):
    logger = logging.getLogger(__name__)
    boxes, logits, phrases = predict(
        dino_model, image, prompt, box_threshold, text_threshold, device
    )
    logger.info(f"Predicted {len(boxes)} bounding boxes")
    return boxes, logits, phrases


def predict_masks(sam_predictor, image_source, boxes, device):
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
    # get first mask since batch size is 1
    mask = masks[0]
    logger.info(f"Flattened masks to shape {mask.shape}")

    mask, _ = remove_small_regions(mask, 100 * 100, "holes")
    mask, _ = remove_small_regions(mask, 100 * 100, "islands")
    logger.info(f"Removed small regions from mask")

    return mask


def convert_mask_to_rle(mask):
    logger = logging.getLogger(__name__)
    rle = coco_mask.encode(np.asfortranarray(mask))
    logger.info(f"Converted mask to RLE")
    return rle


def rle_to_serializable(rle):
    logger = logging.getLogger(__name__)
    serializable_rle = rle.copy()
    # Convert 'counts' from bytes to base64 encoded string for JSON serialization
    serializable_rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
    logger.info(f"Converted RLE to serializable format")
    return serializable_rle


def serializable_to_rle(serializable_rle):
    logger = logging.getLogger(__name__)
    rle = serializable_rle.copy()
    # Convert 'counts' from base64 encoded string back to bytes
    rle['counts'] = base64.b64decode(serializable_rle['counts'].encode('utf-8'))
    logger.info(f"Converted serializable RLE to RLE")
    return rle


def save_mask_as_rle(rle, output_dir, filename):
    logger = logging.getLogger(__name__)
    serializable_rle = rle_to_serializable(rle)
    serialized_json = json.dumps(serializable_rle)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(serialized_json)
    logger.info(f"Saved RLE to {output_path}")


def load_rle_from_file(filename):
    logger = logging.getLogger(__name__)
    with open(filename, 'r') as f:
        serializable_rle = json.load(f)
    logger.info(f"Loaded RLE from {filename}")
    return serializable_to_rle(serializable_rle)


def initialize_models(device):
    logger = logging.getLogger(__name__)
    sam_checkpoint = "/scratch/gpfs/eh0560/segment-anything/sam_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_predictor = load_sam_model(sam_checkpoint, model_type, device)

    dino_model_path = "/scratch/gpfs/eh0560/GroundingDINO/models/groundingdino_swinb_cogcoor.pth"
    dino_config_path = "/scratch/gpfs/eh0560/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    dino_model = load_dino_model(dino_config_path, dino_model_path, device)

    logger.info(f"Initialized models")
    return sam_predictor, dino_model


def process_image(image_path, sam_predictor, dino_model, device, text_prompt, box_threshold, text_threshold):
    logger = logging.getLogger(__name__)
    image_source, image = load_image(image_path)
    boxes, _, _ = predict_bounding_boxes(dino_model, image, device, text_prompt, box_threshold, text_threshold)
    mask = predict_masks(sam_predictor, image_source, boxes, device)
    rle = convert_mask_to_rle(mask)
    logger.info(f"Processed image {image_path}")
    return rle


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting segmentation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor, dino_model = initialize_models(device)

    text_prompt = "scanned document"
    box_threshold = 0.45
    text_threshold = 0.25

    image_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/images"
    image_paths = get_images_from_dir(image_dir)
    image_paths.sort()

    masks_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/document_masks"

    for image_path in image_paths:
        try:
            image_directory = os.path.dirname(image_path).split("/")[-1]
            rle_mask = process_image(image_path, sam_predictor, dino_model, device, text_prompt, box_threshold,
                                     text_threshold)

            output_dir = os.path.join(masks_dir, image_directory)
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path).replace(".jpg", ".json")
            save_mask_as_rle(rle_mask, output_dir, filename)
        except Exception as e:
            logger.error(f"Failed to process image {image_path} with error {e}")
            continue

    logger.info(f"Finished segmentation")

if __name__ == '__main__':
    main()
