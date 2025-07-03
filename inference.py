import os
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import argparse
import cv2
from transformers import pipeline as hf_pipeline
from layout_generator import DinoSamGrounding

def run_inference(args: argparse.Namespace) -> None:
    """
    Executes the virtual staging pipeline:
    1. Generates a pseudo-staged image using ControlNet guidance.
    2. Extracts a robust mask using GroundingDINO+SAM.
    3. Performs high-quality inpainting with the generated mask.
    Saves all intermediate and final results to the specified output directory.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    device: str = "cuda"

    controlnet_inpaint: ControlNetModel = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    controlnet_canny: ControlNetModel = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    controlnet_depth: ControlNetModel = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

    pipeline: StableDiffusionControlNetInpaintPipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_inpaint, controlnet_canny, controlnet_depth],
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    depth_estimator = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-base-hf", device=device)
    layout_generator: DinoSamGrounding = DinoSamGrounding(device=device)
    layout_generator.load()

    empty_image: Image.Image = Image.open(args.input_image).convert("RGB").resize((512, 512))
    canny_image_np: np.ndarray = cv2.Canny(np.array(empty_image), 100, 200)
    canny_image: Image.Image = Image.fromarray(canny_image_np)
    depth_map = depth_estimator(empty_image)['depth']
    depth_image: Image.Image = depth_map.convert("RGB")

    generator = torch.manual_seed(args.seed)
    control_images: list[Image.Image] = [empty_image, canny_image, depth_image]
    controlnet_conditioning_scale: list[float] = [1.0, 0.3, 0.3]

    pseudo_staged_image: Image.Image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=empty_image,
        mask_image=Image.new('L', (512, 512), 255),
        control_image=control_images,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=30,
        generator=generator,
        guidance_scale=9.5
    ).images[0]

    FURNITURE_QUERY: str = "furniture . sofa . chair . table . lamp . rug . plant . decor . art"
    layout_mask_np: np.ndarray = layout_generator.generate_mask_from_text(
        pseudo_staged_image,
        text_prompt=FURNITURE_QUERY,
        box_threshold=args.box_threshold
    )

    if np.sum(layout_mask_np) == 0:
        layout_mask_np[0:5, 0:5] = 255

    layout_mask: Image.Image = Image.fromarray(layout_mask_np)

    generator = torch.manual_seed(args.seed)
    final_control_images: list[Image.Image] = [empty_image, canny_image, depth_image]
    final_conditioning_scale: list[float] = [1.0, 0.1, 0.1]

    result_image: Image.Image = pipeline(
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

    layout_generator.release()

    empty_image.save(os.path.join(args.output_dir, "0_empty_original.png"))
    ImageOps.invert(canny_image).save(os.path.join(args.output_dir, "1a_canny_guidance.png"))
    depth_image.save(os.path.join(args.output_dir, "1b_depth_guidance.png"))
    pseudo_staged_image.save(os.path.join(args.output_dir, "2_pseudo_staged.png"))
    layout_mask.save(os.path.join(args.output_dir, "3_generated_mask.png"))
    result_image.save(os.path.join(args.output_dir, "4_final_result.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robust inference for virtual staging.")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to the ControlNet inpaint model.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/inference_output", help="Directory to save outputs.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad lighting, ugly, deformed, blurry, watermark, text, signature", help="Negative prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Confidence threshold for GroundingDINO object detection.")
    args: argparse.Namespace = parser.parse_args()
    run_inference(args)
