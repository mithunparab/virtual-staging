import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import gc
import numpy as np
import cv2
from PIL import Image

class FlorenceModel:
    """A wrapper to load and release the Florence-2 model cleanly."""
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading Florence-2 model onto {self.device}...")
        model_id = 'microsoft/Florence-2-large'
        # Using float16 and flash_attention_2 for speed and memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Florence-2 model loaded.")

    def release(self):
        print(f"Releasing Florence-2 model from {self.device}...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model released and cache cleared.")

    def run_inference_single(self, image: Image.Image, task_prompt: str):
        """Runs inference for a SINGLE image and a SINGLE task prompt string."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")
        
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # Use task_prompt for post-processing
        return self.processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)


def get_caption(model_wrapper: FlorenceModel, image: Image.Image) -> str:
    """Generates a detailed caption for a single image."""
    task = '<MORE_DETAILED_CAPTION>'
    try:
        parsed = model_wrapper.run_inference_single(image, task)
        # The result for this task is a dictionary with the caption
        return parsed.get(task, [""])[0]
    except Exception as e:
        print(f"Error during captioning: {e}")
        return "A room with furniture." # Return a safe default

def get_mask_from_regions(model_wrapper: FlorenceModel, image: Image.Image) -> np.ndarray:
    """
    Generates a compound mask by proposing regions and filtering out background.
    This is the new, robust method for intelligent furniture segmentation.
    """
    task = '<DENSE_REGION_CAPTION>'
    
    h, w = image.size[1], image.size[0]
    compound_mask = np.zeros((h, w), dtype=np.uint8)

    # Keywords to identify non-furniture background items
    background_keywords = ['wall', 'floor', 'ceiling', 'window', 'door', 'rug', 'carpet', 'curtain', 'light', 'shadow']

    try:
        results = model_wrapper.run_inference_single(image, task)
        # The result is a dictionary with a list of bounding boxes and captions
        regions = results.get(task, [])
        
        for region in regions:
            caption = ' '.join(region['labels']).lower()
            
            # Check if the caption contains any background keywords
            if any(keyword in caption for keyword in background_keywords):
                continue
            
            # Use the polygon for the mask if available, otherwise fall back to bbox
            if 'polygons' in region and region['polygons']:
                for poly in region['polygons']:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(compound_mask, [pts], 255)
            elif 'bboxes' in region and region['bboxes']:
                 for bbox in region['bboxes']:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(compound_mask, (x1, y1), (x2, y2), 255, -1)

    except Exception as e:
        print(f"CRITICAL ERROR during region proposal: {e}")
        # Return an empty mask on failure to avoid masking the whole image
        return np.zeros((h, w), dtype=np.uint8)

    # Dilate to ensure complete coverage and connect nearby parts
    return cv2.dilate(compound_mask, np.ones((5,5), np.uint8), iterations=3)