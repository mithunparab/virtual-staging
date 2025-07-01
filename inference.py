import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import cv2
import argparse

from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

def generate_mask_from_image(image_pil, models):
    """
    Generates a furniture mask for a single PIL image using a pre-loaded model.
    """
    processor = models['processor']
    model = models['model']
    device = models['device']

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image_pil.size[::-1]])[0]
    
    furniture_ids = [8, 13, 16, 18, 22, 28, 30, 32, 53] 
    
    mask = torch.zeros_like(result, dtype=torch.uint8)
    for furniture_id in furniture_ids:
        mask[result == furniture_id] = 255
        
    final_mask_np = mask.cpu().numpy()
    
    return cv2.dilate(final_mask_np, np.ones((5,5),np.uint8), iterations=3)

def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("Loading all required models...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled.")
    except Exception:
        print("xformers not available.")

    print("Loading segmentation model...")
    segmentation_models = {
        'processor': AutoImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade"),
        'model': MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade").to(device),
        'device': device
    }
    print("Models loaded successfully.")

    empty_image = Image.open(args.input_image).convert("RGB").resize((512, 512))

    print("\nStep 1: Generating a low-quality pseudo-staged image for layout...")
    generator = torch.manual_seed(args.seed)
    pseudo_staged_image = pipeline(
        prompt=args.prompt,
        image=empty_image,
        mask_image=Image.new('L', (512, 512), 255),
        control_image=empty_image,
        num_inference_steps=20,
        generator=generator,
        guidance_scale=7.0,
    ).images[0]
    
    print("Step 2: Generating a layout mask from the pseudo-staged image...")
    layout_mask_np = generate_mask_from_image(pseudo_staged_image, segmentation_models)
    layout_mask = Image.fromarray(layout_mask_np)

    print("Step 3: Running final high-quality inpainting with the generated mask...")
    generator = torch.manual_seed(args.seed)
    result_image = pipeline(
        prompt=args.prompt,
        image=empty_image,
        mask_image=layout_mask,
        control_image=empty_image,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5,
    ).images[0]

    print(f"\nInference complete. Saving results to {args.output_dir}")
    empty_image.save(os.path.join(args.output_dir, "0_empty_original.png"))
    pseudo_staged_image.save(os.path.join(args.output_dir, "1_pseudo_staged.png"))
    layout_mask.save(os.path.join(args.output_dir, "2_generated_mask.png"))
    result_image.save(os.path.join(args.output_dir, "3_final_result.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run virtual staging inference.")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to the fine-tuned ControlNet model directory.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the empty room image.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/inference_results", help="Directory to save the output images.")
    # THIS IS THE CORRECTED LINE
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired staged scene.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    run_inference(args)