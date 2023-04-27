# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import imutils
from typing import List
import numpy as np
import cv2
from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import ast

os.system(
    "wget - q https: // dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask_limit: int = Input(
            default=None, description="maximum number of masks to return. If -1 or None, all masks will be returned. NOTE: The masks are sorted by predicted_iou."),
        mask_only: bool = Input(
            default=False, description="If True, the output will only include mask."),
        points_per_side: int = Input(
            default=32, description="The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, point_grids must provide explicit point sampling."),
        pred_iou_thresh: float = Input(
            default=0.88, description="A filtering threshold in [0,1], using the model's predicted mask quality."),
        stability_score_thresh: float = Input(
            default=0.95, description="A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions."),
        stability_score_offset: float = Input(
            default=1.0, description="The amount to shift the cutoff when calculated the stability score."),
        box_nms_thresh: float = Input(
            default=0.7, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks."),
        crop_n_layers: int = Input(
            default=0, description="If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops"),
        crop_nms_thresh: float = Input(
            default=0.7, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops."),
        crop_overlap_ratio: float = Input(
            default=512 / 1500, description="Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap."),
        crop_n_points_downscale_factor: int = Input(
            default=1, description="The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n."),
        min_mask_region_area: int = Input(
            default=0, description="If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area."),
    ) -> List[Path]:
        args = locals()
        del args["self"]
        del args["image"]
        del args["mask_limit"]
        del args["mask_only"]

        # setups the predictor and image
        mask_predictor = SamAutomaticMaskGenerator(self.sam, **args)
        image_bgr = cv2.imread(str(image))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # generate the masks
        sam_result = mask_predictor.generate(image_rgb)

        # SamAutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information about the mask:
        # - segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        # - area - [int] - the area of the mask in pixels
        # - bbox - [List[int]] - the boundary box of the mask in xywh format
        # - predicted_iou - [float] - the model's own prediction for the quality of the mask
        # - point_coords - [List[List[float]]] - the sampled input point that generated this mask
        # - stability_score - [float] - an additional measure of mask quality
        # - crop_box - List[int] - the crop of the image used to generate this mask in xywh format

        # sort them by `predicted_iou`
        masks = [mask['segmentation'] for mask in sorted(
            sam_result, key=lambda x: x['predicted_iou'], reverse=True)]

        # save the masks + mased to files
        mask_images = []
        masked_images = []
        for i, mask in enumerate(masks[:mask_limit]):
            # create a binary mask from the boolean array
            mask_image = np.uint8(mask) * 255

            # save the mask image to a file
            mask_filename = f"mask_{i}.png"
            cv2.imwrite(mask_filename, mask_image)
            mask_images.append(Path(mask_filename))

            if not mask_only:
                # apply the mask to the image
                b, g, r = cv2.split(image_bgr)
                masked_image = cv2.merge([b, g, r, mask_image], 4)

                # save the masked image to a file
                masked_filename = f"masked_{i}.png"
                cv2.imwrite(masked_filename, masked_image)
                masked_images.append(Path(masked_filename))

        return mask_images + masked_images
