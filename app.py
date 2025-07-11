# %%writefile app.py
import os
import torch
from PIL import Image, ImageOps
import numpy as np
import cv2
import gradio as gr
import gc
import sys
import traceback
from datetime import datetime

APP_ROOT = "/kaggle/working/virtual-staging"
OUTPUT_DIR = os.path.join(APP_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Output directory set to: {OUTPUT_DIR} ---")
GROUNDING_DINO_LOCAL_PATH = os.path.join(APP_ROOT, "groundingdino_local")
if os.path.exists(GROUNDING_DINO_LOCAL_PATH) and GROUNDING_DINO_LOCAL_PATH not in sys.path:
    sys.path.insert(0, GROUNDING_DINO_LOCAL_PATH)
    print(f"✅ Added vendorized GroundingDINO to PYTHONPATH: {GROUNDING_DINO_LOCAL_PATH}")
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from transformers import pipeline as hf_pipeline
try:
    from groundingdino.util.inference import load_model as load_gdino_model, predict as predict_gdino
    import groundingdino.datasets.transforms as T
except ImportError as e: raise e
HF_USERNAME = "Nightfury16"
BASE_SD_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_INPAINT_REPO = f"{HF_USERNAME}/virtual-staging-controlnet"
CONTROLNET_CANNY_REPO = "lllyasviel/control_v11p_sd15_canny"
CONTROLNET_DEPTH_REPO = "lllyasviel/sd-controlnet-depth"
LORA_MODEL_REPO = f"{HF_USERNAME}/virtual-staging-lora-sd-v1-5"
SAM_CHECKPOINT = os.path.join(APP_ROOT, "weights/sam_l.pt")
GROUNDING_DINO_CONFIG = os.path.join(APP_ROOT, "groundingdino_local/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(APP_ROOT, "weights/groundingdino_swint_ogc.pth")
DEVICE, DTYPE = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)

def box_cxcywh_to_xyxy(x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if x.nelement() == 0: return x
    x_c, y_c, w, h = x.unbind(1); b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b = torch.stack(b, dim=1); b[:, [0, 2]] *= width; b[:, [1, 3]] *= height; return b
def resize_and_pad(image: Image.Image, target_size: tuple[int, int], background_color: tuple[int, int, int] = (0, 0, 0)) -> tuple[Image.Image, tuple[int, int, int, int]]:
    original_width, original_height = image.size; target_width, target_height = target_size
    ratio_w, ratio_h = target_width / original_width, target_height / original_height
    if ratio_w < ratio_h: new_width, new_height = target_width, round(original_height * ratio_w)
    else: new_height, new_width = target_height, round(original_width * ratio_h)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, background_color)
    paste_x, paste_y = (target_width - new_width) // 2, (target_height - new_height) // 2
    new_image.paste(image, (paste_x, paste_y)); crop_box = (paste_x, paste_y, paste_x + new_width, paste_y + new_height)
    return new_image, crop_box
class SAMModel:
    def __init__(self, device: str = 'cuda:0'): self.device, self.model = device, None
    def load(self, model_path: str = SAM_CHECKPOINT):
        from ultralytics import SAM; print(f"Loading SAM model from: {model_path}..."); self.model = SAM(model_path).to(self.device); print("SAM loaded.")
    def segment_from_boxes(self, image: Image.Image, bboxes: torch.Tensor) -> np.ndarray:
        if self.model is None: raise RuntimeError("SAM Model not loaded.")
        if bboxes.nelement() == 0: return np.zeros((image.height, image.width), dtype=np.uint8)
        results = self.model(image, bboxes=bboxes, verbose=False)
        if not results or not results[0].masks: return np.zeros((image.height, image.width), dtype=np.uint8)
        final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for mask_data in results[0].masks.data: final_mask = np.maximum(final_mask, mask_data.cpu().numpy().astype(np.uint8) * 255)
        return final_mask
class DinoSamGrounding:
    def __init__(self, device: str = 'cuda:0'):
        if predict_gdino is None: raise ImportError("GroundingDINO not accessible.")
        self.device, self.grounding_dino_model, self.sam_wrapper = device, None, SAMModel(device=device)
    def load(self, config_path: str = GROUNDING_DINO_CONFIG, checkpoint_path: str = GROUNDING_DINO_CHECKPOINT):
        print("Loading GroundingDINO model..."); self.grounding_dino_model = load_gdino_model(config_path, checkpoint_path, device=self.device); self.sam_wrapper.load(); print("GroundingDINO and SAM loaded.")
    def generate_mask_from_text(self, image: Image.Image, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> np.ndarray:
        if self.grounding_dino_model is None: raise RuntimeError("Models not loaded.")
        transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_tensor, _ = transform(image, None)
        boxes_relative, _, _ = predict_gdino(model=self.grounding_dino_model, image=image_tensor, caption=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
        if boxes_relative.nelement() == 0: return np.zeros((image.height, image.width), dtype=np.uint8)
        H, W = image.height, image.width; boxes_absolute = box_cxcywh_to_xyxy(x=boxes_relative, width=W, height=H).to(self.device)
        mask = self.sam_wrapper.segment_from_boxes(image, bboxes=boxes_absolute)
        if np.sum(mask) > 0: mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=3)
        return mask

print("--- Initializing and Pre-loading All Models ---")
global_models = {}
try:
    print("Loading Layout Generator (DINO + SAM)..."); layout_gen = DinoSamGrounding(device=DEVICE); layout_gen.load(); global_models["layout_generator"] = layout_gen; print("✅ Layout Generator loaded.")
    print("Loading Depth Estimator..."); global_models["depth_estimator"] = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-base-hf", device=DEVICE); print("✅ Depth Estimator loaded.")
    print("Loading ControlNets..."); controlnet_inpaint = ControlNetModel.from_pretrained(CONTROLNET_INPAINT_REPO, torch_dtype=DTYPE); controlnet_canny = ControlNetModel.from_pretrained(CONTROLNET_CANNY_REPO, torch_dtype=DTYPE); controlnet_depth = ControlNetModel.from_pretrained(CONTROLNET_DEPTH_REPO, torch_dtype=DTYPE); print("✅ ControlNets loaded.")
    print("Loading and configuring main Stable Diffusion pipeline..."); pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(BASE_SD_MODEL, controlnet=[controlnet_inpaint, controlnet_canny, controlnet_depth], torch_dtype=DTYPE, safety_checker=None).to(DEVICE); pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config); global_models["main_pipeline"] = pipeline; print("✅ Main pipeline loaded.")
    print("--- All models loaded. Launching Gradio UI. ---")
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}"); global_models["loading_error"] = str(e)


def run_virtual_staging(
    input_image: Image.Image, prompt: str, negative_prompt: str, use_canny: bool, use_depth: bool, use_lora: bool, seed: int, progress=gr.Progress()
):
    try:
        if input_image is None:
            raise gr.Error("Please upload an image or select an example before generating.")

        if "loading_error" in global_models: raise gr.Error(f"A model failed to load at startup: {global_models['loading_error']}")
        pipeline = global_models["main_pipeline"]; depth_estimator = global_models["depth_estimator"]; layout_generator = global_models["layout_generator"]

        if seed == -1 or seed is None: seed = np.random.randint(0, 2**32 - 1)
        print(f"--- Using Seed: {seed} ---"); generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
        os.makedirs(run_output_dir, exist_ok=True); input_image.save(os.path.join(run_output_dir, "00_input.png"))
        
        if use_lora:
            progress(0, desc="Loading LoRA weights..."); pipeline.load_lora_weights(LORA_MODEL_REPO, subfolder="checkpoint-3000", weight_name="pytorch_lora_weights.safetensors")
        padded_image, crop_box = resize_and_pad(input_image.convert("RGB"), (1024, 1024))
        canny_image = Image.fromarray(cv2.Canny(np.array(padded_image), 100, 200)) if use_canny else None
        if canny_image: canny_image.save(os.path.join(run_output_dir, "01_control_canny.png"))
        depth_image = depth_estimator(padded_image)['depth'].convert("RGB") if use_depth else None
        if depth_image: depth_image.save(os.path.join(run_output_dir, "02_control_depth.png"))
        progress(0.2, desc="Phase 1/3: Generating layout concept..."); phase1_control_images, phase1_scales = [padded_image], [0.0] if (use_canny or use_depth) else [1.0]
        if use_canny: phase1_control_images.append(canny_image); phase1_scales.append(0.3)
        if use_depth: phase1_control_images.append(depth_image); phase1_scales.append(0.3)
        pseudo_staged = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=padded_image, mask_image=Image.new('L', (1024, 1024), 255), control_image=phase1_control_images, controlnet_conditioning_scale=phase1_scales, num_inference_steps=30, guidance_scale=9.5, generator=generator).images[0]
        pseudo_staged.save(os.path.join(run_output_dir, "03_pseudo_staged.png"))
        progress(0.5, desc="Phase 2/3: Analyzing layout..."); layout_mask_np = layout_generator.generate_mask_from_text(pseudo_staged, "furniture . sofa . chair . table . lamp . rug . plant . decor . art", 0.3)
        layout_mask = Image.fromarray(layout_mask_np) if np.sum(layout_mask_np) > 0 else Image.new('L', (1024, 1024), 255)
        layout_mask.save(os.path.join(run_output_dir, "04_layout_mask.png"))
        agnostic_image = Image.composite(Image.new('RGB', padded_image.size), padded_image, ImageOps.invert(layout_mask.convert('L')))
        agnostic_image.save(os.path.join(run_output_dir, "05_agnostic_image.png"))
        progress(0.6, desc="Phase 3/3: Final Inpainting..."); final_control_images, final_scales = [agnostic_image], [1.0]
        if use_canny: final_control_images.append(canny_image); final_scales.append(0.1)
        if use_depth: final_control_images.append(depth_image); final_scales.append(0.1)
        final_padded = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=padded_image, mask_image=layout_mask, control_image=final_control_images, controlnet_conditioning_scale=final_scales, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        if use_lora: pipeline.unload_lora_weights()
        final_cropped = final_padded.crop(crop_box)
        final_cropped.save(os.path.join(run_output_dir, "06_final_result.png"))
        output_gallery = [(os.path.join(run_output_dir, f), f.split('_', 1)[1][:-4].replace('_', ' ').title()) for f in sorted(os.listdir(run_output_dir)) if f.endswith('.png') and not f.startswith('00_')]
        return {final_image_output: final_cropped, gallery_output: output_gallery, seed_input: seed}

    except Exception as e:
        error_message = traceback.format_exc(); print(f"!!! AN ERROR OCCURRED !!!\n{error_message}")
        raise gr.Error(f"An error occurred: {e}")

with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# Virtual Staging AI")
    gr.Markdown("All models are pre-loaded. Configure your generation and click 'Generate Staging'.")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Empty Room Image")
            prompt = gr.Textbox(label="Staging Prompt", placeholder="e.g., 'A cozy living room...'", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad lighting, ugly, deformed, blurry, watermark, text, signature", lines=3)
            with gr.Accordion("Model Configuration", open=True):
                with gr.Row():
                    use_canny = gr.Checkbox(label="Use Canny Edge", value=True)
                    use_depth = gr.Checkbox(label="Use Depth Map", value=True)
                    use_lora = gr.Checkbox(label="Use Staging LoRA", value=True)
                seed_input = gr.Number(label="Seed", value=-1, info="Use -1 for a random seed.", precision=0, interactive=True)
            submit_btn = gr.Button("Generate Staging", variant="primary")
        with gr.Column(scale=1):
            final_image_output = gr.Image(label="Final Staged Image", type="pil")
            gallery_output = gr.Gallery(label="All Generated Steps", show_label=True, columns=3, height="auto")
    
    submit_btn.click(
        fn=run_virtual_staging,
        inputs=[input_image, prompt, negative_prompt, use_canny, use_depth, use_lora, seed_input],
        outputs=[final_image_output, gallery_output, seed_input]
    )
    
    gr.Examples(
        examples=[
            ["example_images/empty_room_1.png", "A sleek, open-concept modern kitchen bathed in natural light, featuring matte black cabinetry, marble countertops, and minimalist pendant lighting."],
            ["example_images/empty_room_2.png", "Add a small wooden study table with a comfortable chair, a desk lamp, and subtle decor or framed artwork."]
        ],
        inputs=[input_image, prompt, negative_prompt, use_canny, use_depth, use_lora, seed_input]
    )

demo.queue().launch(debug=True, share=True)