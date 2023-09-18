import glob
import json
import os
import base64
import torch
import numpy as np
from pycocotools import mask as coco_mask
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions


def load_sam_model(checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def load_dino_model(config_path, checkpoint_path, device):
    model = load_model(config_path, checkpoint_path, device)
    return model


def get_images_from_dir(image_dir):
    image_paths = glob.glob(image_dir + "/*/*.jpg")
    return image_paths


def predict_bounding_boxes(
        dino_model, image, device, prompt, box_threshold=0.45, text_threshold=0.25
):
    boxes, logits, phrases = predict(
        dino_model, image, prompt, box_threshold, text_threshold, device
    )
    return boxes, logits, phrases


def predict_masks(sam_predictor, image_source, boxes, device):
    sam_predictor.set_image(image_source)

    height, width, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(device)

    masks, iou_preds, logits = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    flattened_mask = masks.sum(dim=1)
    masks = (flattened_mask == True).cpu().numpy()
    # get first mask since batch size is 1
    mask = masks[0]

    mask, _ = remove_small_regions(mask, 100 * 100, "holes")
    mask, _ = remove_small_regions(mask, 100 * 100, "islands")

    return mask


def convert_mask_to_rle(mask):
    rle = coco_mask.encode(np.asfortranarray(mask))
    return rle


def rle_to_serializable(rle):
    serializable_rle = rle.copy()
    # Convert 'counts' from bytes to base64 encoded string for JSON serialization
    serializable_rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
    return serializable_rle


def serializable_to_rle(serializable_rle):
    rle = serializable_rle.copy()
    # Convert 'counts' from base64 encoded string back to bytes
    rle['counts'] = base64.b64decode(serializable_rle['counts'].encode('utf-8'))
    return rle


def save_mask_as_rle(rle, output_dir, filename):
    serializable_rle = rle_to_serializable(rle)
    serialized_json = json.dumps(serializable_rle)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(serialized_json)


def load_rle_from_file(filename):
    with open(filename, 'r') as f:
        serializable_rle = json.load(f)
    return serializable_to_rle(serializable_rle)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_checkpoint = (
        "/scratch/gpfs/eh0560/segment-anything/sam_models/sam_vit_h_4b8939.pth"
    )
    model_type = "vit_h"

    sam_predictor = load_sam_model(sam_checkpoint, model_type, device)

    dino_model_path = (
        "/scratch/gpfs/eh0560/GroundingDINO/models/groundingdino_swinb_cogcoor.pth"
    )
    dino_config_path = "/scratch/gpfs/eh0560/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"

    dino_model = load_dino_model(dino_config_path, dino_model_path, device)

    text_prompt = "scanned document"
    box_threshold = 0.45
    text_threshold = 0.25

    image_paths = get_images_from_dir(
        "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images"
    )
    image_paths.sort()
    image_path = image_paths[0]

    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict_bounding_boxes(
        dino_model, image, device, text_prompt, box_threshold, text_threshold
    )
    mask = predict_masks(sam_predictor, image_source, boxes, device)

    output_dir = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/cudl_images/masks"
    filename = os.path.basename(image_path).replace(".jpg", ".json")
    rle = convert_mask_to_rle(mask)
    save_mask_as_rle(rle, output_dir, filename)
