import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import gc
import numpy as np
import cv2

class FlorenceModel:
    """A wrapper to load and release the Florence-2 model cleanly."""
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading Florence-2 model onto {self.device}...")
        model_id = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded.")

    def release(self):
        print(f"Releasing Florence-2 model from {self.device}...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model released and cache cleared.")

    def run_inference_single(self, image, prompt):
        """Runs inference for a SINGLE image and a SINGLE prompt string."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(generated_text, task=prompt)


def get_caption(model_wrapper, image):
    """Generates a caption for a single image."""
    task = '<CAPTION>'
    parsed = model_wrapper.run_inference_single(image, task)
    return parsed.get(task, "")

def get_mask_from_regions(model_wrapper, image):
    """
    Generates a compound mask by proposing regions and filtering out background.
    THIS IS THE NEW, ROBUST METHOD.
    """
    task = '<MORE_DETAILED_CAPTION>'
    
    h, w = image.height, image.width
    compound_mask = np.zeros((h, w), dtype=np.uint8)

    try:
        res = model_wrapper.run_inference_single(image, task)
        # The result is a dictionary with bounding boxes and captions for each region
        results = res.get(task, [])
        
        for region in results:
            # Filter out regions that are likely background
            caption = region['caption'].lower()
            if 'wall' in caption or 'floor' in caption or 'ceiling' in caption:
                continue
            
            # Use the polygon for the mask
            poly = region.get('polygon')
            if poly:
                scaled_poly = [(int(p[0] * w / 1000), int(p[1] * h / 1000)) for p in poly]
                pts = np.array(scaled_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(compound_mask, [pts], 255)
                
    except Exception as e:
        print(f"CRITICAL ERROR during region proposal: {e}")

    return cv2.dilate(compound_mask, np.ones((5,5),np.uint8), iterations=3)