import os
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import argparse
import cv2
from transformers import pipeline as hf_pipeline

from layout_generator import FlorenceModel, SAMModel, get_grounded_mask

def run_inference(args) -> None:
    """
    Runs the Structural Integrity pipeline for virtual staging using multiple ControlNet models.
    Loads required models, prepares input images, generates a pseudo-staged image, creates a layout mask,
    and performs final inpainting. Saves all intermediate and final results to the specified output directory.

    Args:
        args: Parsed command-line arguments containing paths, prompts, and configuration.
    Returns:
        None
    """
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    controlnet_inpaint = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_inpaint, controlnet_canny, controlnet_depth],
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception:
        print("xformers not available.")

    depth_estimator = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-base-hf", device=device)

    florence_wrapper = FlorenceModel(device=device)
    sam_wrapper = SAMModel(device=device)
    florence_wrapper.load()
    sam_wrapper.load()
    print("All models loaded.")

    empty_image = Image.open(args.input_image).convert("RGB").resize((512, 512))
    canny_image_np = cv2.Canny(np.array(empty_image), 100, 200)
    canny_image = Image.fromarray(canny_image_np)
    depth_map = depth_estimator(empty_image)['depth']
    depth_image = depth_map.convert("RGB")

    print("\nStep 1: Generating a pseudo-staged image with Triple ControlNet Guidance...")
    generator = torch.manual_seed(args.seed)
    control_images = [empty_image, canny_image, depth_image]
    controlnet_conditioning_scale = [1.0, 0.5, 0.5]

    pseudo_staged_image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=empty_image,
        mask_image=Image.new('L', (512, 512), 255),
        control_image=control_images,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=30,
        generator=generator,
        guidance_scale=7.5
    ).images[0]

    print("Step 2: Generating a layout mask using the Grounding pipeline...")
    layout_mask_np, _ = get_grounded_mask(florence_wrapper, sam_wrapper, pseudo_staged_image)

    if np.sum(layout_mask_np) == 0:
        print("Warning: Grounding pipeline did not detect any furniture. The final image may be unchanged.")
        layout_mask_np[0:5, 0:5] = 255

    layout_mask = Image.fromarray(layout_mask_np)

    print("Step 3: Running final high-quality inpainting with the generated mask...")
    generator = torch.manual_seed(args.seed)
    final_control_images = [empty_image, canny_image, depth_image]
    final_conditioning_scale = [1.0, 0.0, 0.0]

    result_image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=empty_image,
        mask_image=layout_mask,
        control_image=final_control_images,
        controlnet_conditioning_scale=final_conditioning_scale,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5
    ).images[0]

    florence_wrapper.release()
    sam_wrapper.release()

    print(f"\nInference complete. Saving results to {args.output_dir}")
    empty_image.save(os.path.join(args.output_dir, "0_empty_original.png"))
    ImageOps.invert(canny_image).save(os.path.join(args.output_dir, "1a_canny_guidance.png"))
    depth_image.save(os.path.join(args.output_dir, "1b_depth_guidance.png"))
    pseudo_staged_image.save(os.path.join(args.output_dir, "2_pseudo_staged.png"))
    layout_mask.save(os.path.join(args.output_dir, "3_generated_mask.png"))
    result_image.save(os.path.join(args.output_dir, "4_final_result.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the Structural Integrity pipeline.")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/inference_results")
    parser.add_argument("--prompt", type=str, required=True, help="modern interior decor with detailed furniture and styling, Scandinavian minimalism, neutral tones, no change to room structure or lighting")
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad quality, bad lighting, ugly, deformed, blurry, watermark, text, signature, trees, plants in strange places", help="What to avoid in the generated image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_inference(args)
