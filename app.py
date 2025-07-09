import os
import torch
from PIL import Image, ImageOps
import numpy as np
import cv2
import gradio as gr
import gc
import sys

GROUNDING_DINO_PATH = "./GroundingDINO"
if os.path.exists(GROUNDING_DINO_PATH) and GROUNDING_DINO_PATH not in sys.path:
    sys.path.insert(0, GROUNDING_DINO_PATH)

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from transformers import pipeline as hf_pipeline

try:
    from groundingdino.util.inference import load_model as load_gdino_model, predict as predict_gdino
    import groundingdino.datasets.transforms as T
except ImportError:
    load_gdino_model, predict_gdino, T = None, None, None

def box_cxcywh_to_xyxy(x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Convert bounding boxes from center-x, center-y, width, height format to x1, y1, x2, y2 format.
    """
    if x.nelement() == 0:
        return x
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b = torch.stack(b, dim=1)
    b[:, [0, 2]] *= width
    b[:, [1, 3]] *= height
    return b

class SAMModel:
    """
    Wrapper for Segment Anything Model (SAM) for segmentation from bounding boxes.
    """
    def __init__(self, device: str = 'cuda:0') -> None:
        self.device: str = device
        self.model = None

    def load(self, model_path: str = './weights/sam_l.pt') -> None:
        from ultralytics import SAM
        self.model = SAM(model_path).to(self.device)

    def release(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def segment_from_boxes(self, image: Image.Image, bboxes: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SAM Model not loaded.")
        if bboxes.nelement() == 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        results = self.model(image, bboxes=bboxes, verbose=False)
        if not results or not results[0].masks:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for mask_data in results[0].masks.data:
            final_mask = np.maximum(final_mask, mask_data.cpu().numpy().astype(np.uint8) * 255)
        return final_mask

class DinoSamGrounding:
    """
    Combines GroundingDINO and SAM for text-guided object segmentation.
    """
    def __init__(self, device: str = 'cuda:0') -> None:
        if predict_gdino is None:
            raise ImportError("GroundingDINO is not installed or accessible.")
        self.device: str = device
        self.grounding_dino_model = None
        self.sam_wrapper = SAMModel(device=device)

    def load(self, config_path: str = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", checkpoint_path: str = "./weights/groundingdino_swint_ogc.pth") -> None:
        self.grounding_dino_model = load_gdino_model(config_path, checkpoint_path, device=self.device)
        self.sam_wrapper.load()

    def release(self) -> None:
        if self.grounding_dino_model is not None:
            del self.grounding_dino_model
            self.grounding_dino_model = None
        self.sam_wrapper.release()
        gc.collect()
        torch.cuda.empty_cache()

    def generate_mask_from_text(self, image: Image.Image, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> np.ndarray:
        """
        Generate a segmentation mask for objects matching the text prompt.
        """
        if self.grounding_dino_model is None:
            raise RuntimeError("Models not loaded. Call .load() first.")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor, _ = transform(image, None)
        boxes_relative, logits, phrases = predict_gdino(
            model=self.grounding_dino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        if boxes_relative.nelement() == 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        H, W = image.height, image.width
        boxes_absolute = box_cxcywh_to_xyxy(x=boxes_relative, width=W, height=H)
        boxes_absolute = boxes_absolute.to(self.device)
        mask = self.sam_wrapper.segment_from_boxes(image, bboxes=boxes_absolute)
        if np.sum(mask) > 0:
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
        return mask

HF_USERNAME: str = "Nightfury16"
BASE_SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
CONTROLNET_INPAINT_REPO: str = f"{HF_USERNAME}/virtual-staging-controlnet"
CONTROLNET_CANNY_REPO: str = "lllyasviel/control_v11p_sd15_canny"
CONTROLNET_DEPTH_REPO: str = "lllyasviel/sd-controlnet-depth"
LORA_MODEL_REPO: str = f"{HF_USERNAME}/virtual-staging-lora-sd-v1-5"

GROUNDING_DINO_CONFIG: str = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT: str = "./weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT: str = "./weights/sam_l.pt"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

try:
    controlnet_inpaint = ControlNetModel.from_pretrained(CONTROLNET_INPAINT_REPO, torch_dtype=DTYPE)
    controlnet_canny = ControlNetModel.from_pretrained(CONTROLNET_CANNY_REPO, torch_dtype=DTYPE)
    controlnet_depth = ControlNetModel.from_pretrained(CONTROLNET_DEPTH_REPO, torch_dtype=DTYPE)

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_SD_MODEL,
        controlnet=[controlnet_inpaint, controlnet_canny, controlnet_depth],
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    try:
        pipeline.load_lora_weights(
            LORA_MODEL_REPO,
            weight_name="pytorch_lora_weights.safetensors"
        )
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    depth_estimator = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-base-hf", device=DEVICE)
    layout_generator = DinoSamGrounding(device=DEVICE)
    layout_generator.load(
        config_path=GROUNDING_DINO_CONFIG,
        checkpoint_path=GROUNDING_DINO_CHECKPOINT
    )
except Exception as e:
    print(f"FATAL ERROR during model initialization: {e}")
    pipeline, depth_estimator, layout_generator = None, None, None

def predict_staged_image(input_image: Image.Image, prompt: str) -> Image.Image:
    """
    Perform virtual staging inference given an empty room image and a prompt.

    Args:
        input_image (Image.Image): Input empty room image.
        prompt (str): Staging prompt.

    Returns:
        Image.Image: Virtually staged image.
    """
    if pipeline is None:
        return Image.new('RGB', (512, 512), color='red')

    empty_image: Image.Image = input_image.convert("RGB").resize((1024, 1024))
    canny_image_np: np.ndarray = cv2.Canny(np.array(empty_image), 100, 200)
    canny_image: Image.Image = Image.fromarray(canny_image_np)
    depth_map = depth_estimator(empty_image)['depth']
    depth_image: Image.Image = depth_map.convert("RGB")

    negative_prompt: str = "low quality, bad lighting, ugly, deformed, blurry, watermark, text, signature"
    generator = torch.manual_seed(42)

    control_images_phase1 = [empty_image, canny_image, depth_image]
    controlnet_conditioning_scale_phase1 = [1.0, 0.3, 0.3]

    pseudo_staged_image: Image.Image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=empty_image,
        mask_image=Image.new('L', (1024, 1024), 255),
        control_image=control_images_phase1,
        controlnet_conditioning_scale=controlnet_conditioning_scale_phase1,
        num_inference_steps=30,
        generator=generator,
        guidance_scale=9.5
    ).images[0]

    FURNITURE_QUERY: str = "furniture . sofa . chair . table . lamp . rug . plant . decor . art"
    BOX_THRESHOLD: float = 0.3

    layout_mask_np: np.ndarray = layout_generator.generate_mask_from_text(
        pseudo_staged_image,
        text_prompt=FURNITURE_QUERY,
        box_threshold=BOX_THRESHOLD
    )

    if np.sum(layout_mask_np) == 0:
        layout_mask_np[0:5, 0:5] = 255

    layout_mask: Image.Image = Image.fromarray(layout_mask_np)

    final_control_images = [empty_image, canny_image, depth_image]
    final_conditioning_scale = [1.0, 0.1, 0.1]

    result_image: Image.Image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=empty_image,
        mask_image=layout_mask,
        control_image=final_control_images,
        controlnet_conditioning_scale=final_conditioning_scale,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5
    ).images[0]

    return result_image

description_content: str = """
This project leverages a powerful pipeline of generative AI models to perform virtual staging on empty room images.
The primary goal is to create high-quality, photorealistic staged interior designs while meticulously preserving the original room's structural integrity and 3D geometry.

### Approach
Our approach is built around a synergistic, multi-stage process, where each component is chosen for its specific strengths:

1.  **Creative Layout Generation:** An initial "pseudo-staged" image is generated to populate the room with furniture ideas based on the prompt.
2.  **Text-Guided Masking:** Grounding DINO and SAM identify and precisely segment objects within the pseudo-staged image to create a 'staging area' mask.
3.  **Multi-ControlNet Guided Inpainting:** The final staged image is generated in a single pass, using your trained Inpainting ControlNet, plus Canny and Depth ControlNets, guided by the generated mask, to inject furniture while preserving original room geometry.

---
**Input an empty room image and describe your desired staging style!**
"""

gr.Interface(
    fn=predict_staged_image,
    inputs=[
        gr.Image(type="pil", label="Upload Empty Room Image"),
        gr.Textbox(label="Staging Prompt", placeholder="e.g., 'modern interior styling, add detailed furniture, rugs, indoor plants, wall art, photorealistic materials, soft textures, warm tones'", lines=2)
    ],
    outputs=gr.Image(type="pil", label="Virtually Staged Image"),
    title="Virtual Staging AI",
    description=description_content,
    allow_flagging="never",
    examples=[
        ["./example_images/empty_room_1.png", "A cozy living room with a mid-century modern sofa, a wooden coffee table, and a large abstract painting."],
        ["./example_images/empty_room_2.png", "A luxurious bedroom with a king-sized bed, velvet headboard, and soft, ambient lighting."]
    ]
).launch(debug=True, share=True)
