import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import argparse

# Import the new intelligent layout generator
from layout_generator import FlorenceModel, get_mask_from_regions

def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("Loading Diffusion and ControlNet models...")
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

    # Load the powerful Florence-2 model for segmentation
    florence_wrapper = FlorenceModel(device=device)
    florence_wrapper.load()

    empty_image = Image.open(args.input_image).convert("RGB").resize((512, 512))

    print("\nStep 1: Generating a pseudo-staged image for layout detection...")
    generator = torch.manual_seed(args.seed)
    # Use a full mask to let the model imagine a complete scene
    pseudo_staged_image = pipeline(
        prompt=args.prompt,
        image=empty_image,
        mask_image=Image.new('L', (512, 512), 255),
        control_image=empty_image,
        num_inference_steps=25, # A few more steps can help clarity
        generator=generator,
        guidance_scale=7.0,
    ).images[0]
    
    print("Step 2: Generating a layout mask from the pseudo-staged image using Florence-2...")
    # This is the key improvement: using our robust, region-based segmentation
    layout_mask_np = get_mask_from_regions(florence_wrapper, pseudo_staged_image)
    
    if np.sum(layout_mask_np) == 0:
        print("Warning: Florence-2 did not detect any furniture in the pseudo-staged image. The final image may be unchanged.")
        # Create a tiny mask to avoid errors, but it won't have a real effect
        layout_mask_np[0:5, 0:5] = 255
        
    layout_mask = Image.fromarray(layout_mask_np)

    print("Step 3: Running final high-quality inpainting with the generated mask...")
    generator = torch.manual_seed(args.seed)
    result_image = pipeline(
        prompt=args.prompt,
        image=empty_image,
        mask_image=layout_mask,
        control_image=empty_image, # ControlNet condition remains the empty room
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5,
    ).images[0]
    
    # Clean up Florence-2 from memory
    florence_wrapper.release()

    print(f"\nInference complete. Saving results to {args.output_dir}")
    empty_image.save(os.path.join(args.output_dir, "0_empty_original.png"))
    pseudo_staged_image.save(os.path.join(args.output_dir, "1_pseudo_staged.png"))
    layout_mask.save(os.path.join(args.output_dir, "2_generated_mask.png"))
    result_image.save(os.path.join(args.output_dir, "3_final_result.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run virtual staging inference with Florence-2.")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to the fine-tuned ControlNet model directory.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the empty room image.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/inference_results", help="Directory to save the output images.")
    parser.add_argument("--prompt", type=str, required=True, help="A detailed, theme-based prompt describing the desired furniture, style, and lighting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    run_inference(args)