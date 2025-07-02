import gc
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, dynamic_module_utils
from ultralytics import SAM
from unittest.mock import patch
from typing import Union
import os
import traceback

class SAMModel:
    """
    Wrapper for the SAM model using the 'sam_l.pt' checkpoint for segmentation.
    """
    def __init__(self, device='cuda:0'):
        self.device = device
        self.model = None

    def load(self) -> None:
        """
        Loads the SAM model onto the specified device.
        """
        print(f"Loading SAM model onto {self.device}...")
        self.model = SAM('sam_l.pt').to(self.device)
        print("SAM model loaded.")

    def release(self) -> None:
        """
        Releases the SAM model and clears GPU memory.
        """
        print("Releasing SAM model...")
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def segment_from_boxes(self, image: Image.Image, bboxes: list) -> np.ndarray:
        """
        Generates a segmentation mask from bounding boxes.
        Args:
            image (Image.Image): Input image.
            bboxes (list): List of bounding boxes.
        Returns:
            np.ndarray: Segmentation mask.
        """
        if self.model is None:
            raise RuntimeError("SAM Model not loaded.")
        if not bboxes:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        results = self.model(image, bboxes=bboxes, verbose=False)
        if not results or not results[0].masks:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for mask_data in results[0].masks.data:
            final_mask = np.maximum(final_mask, mask_data.cpu().numpy().astype(np.uint8) * 255)
        return final_mask

_original_get_imports = dynamic_module_utils.get_imports
def _patched_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """
    Patch to remove 'flash_attn' import for Florence-2 compatibility.
    """
    imports = _original_get_imports(filename)
    if str(filename).endswith("/modeling_florence2.py"):
        if "flash_attn" in imports:
            imports.remove("flash_attn")
    return imports

class FlorenceModel:
    """
    Wrapper for Florence-2 model for captioning and grounding.
    """
    def __init__(self, device='cuda:0'):
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Loads the Florence-2 model and processor onto the specified device.
        """
        print(f"Loading Florence-2 model onto {self.device}...")
        model_id = 'microsoft/Florence-2-large'
        with patch("transformers.dynamic_module_utils.get_imports", _patched_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Florence-2 model loaded.")

    def release(self) -> None:
        """
        Releases the Florence-2 model and processor, clearing GPU memory.
        """
        print("Releasing Florence-2 model...")
        del self.model, self.processor
        self.model, self.processor = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def run_inference(self, task: str, text_input: str, image: Image.Image) -> dict:
        """
        Runs inference on the Florence-2 model.
        Args:
            task (str): Task prompt.
            text_input (str): Additional text input.
            image (Image.Image): Input image.
        Returns:
            dict: Model output.
        """
        if self.model is None:
            raise RuntimeError("Florence-2 model not loaded.")
        prompt = task if text_input is None else task + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(generated_text, task=task, image_size=image.size)

    def get_caption(self, image: Image.Image) -> str:
        """
        Generates a detailed caption for an image.
        Args:
            image (Image.Image): Input image.
        Returns:
            str: Generated caption.
        """
        try:
            results = self.run_inference('<MORE_DETAILED_CAPTION>', None, image)
            return results.get('<MORE_DETAILED_CAPTION>', ["a furnished room"])[0]
        except Exception:
            return "a furnished room"

    def get_grounded_bboxes(self, image: Image.Image, caption: str) -> list:
        """
        Finds bounding boxes for phrases in a caption.
        Args:
            image (Image.Image): Input image.
            caption (str): Caption text.
        Returns:
            list: List of bounding boxes.
        """
        try:
            results = self.run_inference('<CAPTION_TO_PHRASE_GROUNDING>', caption, image)
            grounding_results = results.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
            all_bboxes = []
            if 'bboxes' in grounding_results:
                all_bboxes.extend(grounding_results['bboxes'])
            for entry in grounding_results.get('labels', []):
                if isinstance(entry, dict) and 'bboxes' in entry:
                    all_bboxes.extend(entry['bboxes'])
            valid_bboxes = [box for box in all_bboxes if box[2] > box[0] and box[3] > box[1]]
            return valid_bboxes
        except Exception:
            return []

def get_grounded_mask(
    florence_wrapper: FlorenceModel,
    sam_wrapper: SAMModel,
    image: Image.Image
) -> tuple[np.ndarray, str]:
    """
    Grounding pipeline: generates a mask and caption for an image.
    Args:
        florence_wrapper (FlorenceModel): Florence-2 model wrapper.
        sam_wrapper (SAMModel): SAM model wrapper.
        image (Image.Image): Input image.
    Returns:
        tuple[np.ndarray, str]: Segmentation mask and caption.
    """
    try:
        caption = florence_wrapper.get_caption(image)
        if not caption or caption == "a furnished room":
            return np.zeros((image.height, image.width), dtype=np.uint8), "a furnished room"
        bboxes = florence_wrapper.get_grounded_bboxes(image, caption)
        if not bboxes:
            return np.zeros((image.height, image.width), dtype=np.uint8), caption
        mask = sam_wrapper.segment_from_boxes(image, bboxes)
        if np.sum(mask) > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
        return mask, caption
    except Exception as e:
        print(f"  > CRITICAL FAILURE during grounding pipeline. Error: {e}")
        traceback.print_exc()
        return np.zeros((image.height, image.width), dtype=np.uint8), "a furnished room"