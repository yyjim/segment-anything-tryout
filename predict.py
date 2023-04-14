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


def sort_by_indexes(lst, indexes, reverse=False):
    return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x:
            x[0], reverse=reverse)]


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
        box: str = Input(
            default=None, description="Bounding box coordinates [x, y, w, h]. If None, the entire image will be used."),
        multimask_output: bool = Input(
            default=False, description="If True, the output will be a list of masks. If False, the output will be a single mask."),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        # convert the string to a Python list
        try:
            # try to convert the string to a Python list
            array = ast.literal_eval(box)
            boxArray = np.array([
                array[0],
                array[1],
                array[0] + array[2],
                array[1] + array[3]
            ])
        except (SyntaxError, ValueError):
            # if it's not a valid Python expression, use the default array
            boxArray = None

        # setups the predictor and image
        mask_predictor = SamPredictor(self.sam)
        image_bgr = cv2.imread(str(image))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # run the prediction
        mask_predictor.set_image(image_rgb)
        masks, scores, logits = mask_predictor.predict(
            box=boxArray,
            multimask_output=multimask_output
        )

        # sort the masks by score
        sorted_masks = sort_by_indexes(masks, scores, reverse=True)

        print(scores)
        # create mask images
        color = np.array([1, 1, 1])
        results = []

        for index, item in enumerate(sorted_masks):
            filename = f"mask.{index}.png"
            mask = sorted_masks[index]
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = (mask_image * 255).astype(np.uint8)
            cv2.imwrite(filename, mask_image)
            results.append(Path(filename))

        # create the output image by cropping the original image with the first mask,
        output_mask = cv2.imread("mask.0.png")
        output_mask = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image_bgr)
        output_image = cv2.merge([b, g, r, output_mask], 4)
        cv2.imwrite('output.png', output_image)

        results.append(Path('output.png'))
        return results
