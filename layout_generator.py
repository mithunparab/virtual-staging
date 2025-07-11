import gc
import os
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import SAM

FURNITURE_PROMPT = "furniture . sofa . chair . table . bed . cabinet . shelf . lamp . plant . decor . rug"
GENERIC_CAPTION = "a photo of a furnished room, professional photography, high quality, 4k"

try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("Warning: GroundingDINO not found. Please clone the repository and install it.")
    print("Run: git clone https://github.com/IDEA-Research/GroundingDINO.git; cd GroundingDINO; pip install -e .")
    load_model, predict, T = None, None, None

def box_cxcywh_to_xyxy(
    x: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """
    Convert bounding boxes from [center_x, center_y, width, height] (relative)
    to [x1, y1, x2, y2] (absolute pixel values).
    """
    if x.nelement() == 0:
        return x
    x_c, y_c, w, h = x.unbind(1)
    b = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h)
    ]
    b = torch.stack(b, dim=1)
    b[:, [0, 2]] *= width
    b[:, [1, 3]] *= height
    return b

class SAMModel:
    """ Wrapper for the SAM segmentation model. """
    def __init__(self, device: str = 'cuda:0') -> None:
        self.device = device
        self.model = None

    def load(self, model_path: str = 'sam_l.pt') -> None:
        """ Load the SAM model. Assumes 'sam_l.pt' is available for download or locally. """
        print(f"Loading SAM model from {model_path} onto {self.device}...")
        self.model = SAM(model_path).to(self.device)
        print("SAM model loaded.")

    def release(self) -> None:
        """ Release the SAM model and free resources. """
        if self.model is not None:
            print("Releasing SAM model...")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def segment_from_boxes(self, image: Image.Image, bboxes: torch.Tensor) -> np.ndarray:
        """ Generate a segmentation mask from bounding boxes. """
        if self.model is None:
            raise RuntimeError("SAM Model not loaded.")
        if bboxes.nelement() == 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)

        results = self.model(np.array(image), bboxes=bboxes.cpu().numpy(), verbose=False)

        if not results or not results[0].masks:
            return np.zeros((image.height, image.width), dtype=np.uint8)

        final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for mask_data in results[0].masks.data:
            final_mask = np.maximum(final_mask, mask_data.cpu().numpy().astype(np.uint8) * 255)
        return final_mask

class DinoSamGrounding:
    """ Combines GroundingDINO and SAM for text-prompted segmentation. """
    def __init__(self, device: str = 'cuda:0') -> None:
        if predict is None:
            raise ImportError("GroundingDINO is not installed or accessible. Check setup.")
        self.device = device
        self.grounding_dino_model = None
        self.sam_wrapper = SAMModel(device=device)

    def load(
        self,
        config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "weights/groundingdino_swint_ogc.pth"
    ) -> None:
        """
        Load GroundingDINO and SAM models.
        NOTE: Assumes GroundingDINO repo is cloned and weights are downloaded.
        """
        print("Loading GroundingDINO model...")
        self.grounding_dino_model = load_model(config_path, checkpoint_path, device=self.device)
        self.sam_wrapper.load()
        print("GroundingDINO and SAM loaded successfully.")

    def release(self) -> None:
        """ Release GroundingDINO and SAM models and free resources. """
        print("Releasing GroundingDINO and SAM models...")
        if self.grounding_dino_model is not None:
            del self.grounding_dino_model
            self.grounding_dino_model = None
        self.sam_wrapper.release()
        gc.collect()
        torch.cuda.empty_cache()

    def generate_mask_from_text(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        debug: bool = True
    ) -> np.ndarray:
        """ Generate a segmentation mask for objects matching a text prompt. """
        if self.grounding_dino_model is None:
            raise RuntimeError("Models not loaded. Call .load() first.")

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image, None)

        boxes_relative, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if boxes_relative.nelement() == 0:
            print(f"Warning: GroundingDINO found no objects for prompt '{text_prompt}'.")
            return np.zeros((image.height, image.width), dtype=np.uint8)

        if debug: print(f"GroundingDINO found {boxes_relative.shape[0]} objects: {phrases}")

        H, W = image.height, image.width
        boxes_absolute = box_cxcywh_to_xyxy(x=boxes_relative, width=W, height=H)
        boxes_absolute = boxes_absolute.to(self.device)

        mask = self.sam_wrapper.segment_from_boxes(image, bboxes=boxes_absolute)
        if np.sum(mask) > 0:
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)

        return mask

def get_grounded_mask(
    grounding_pipeline: "DinoSamGrounding",
    image: Image.Image,
    debug: bool = True
) -> tuple[np.ndarray, str]:
    """
    Generates a mask for furniture using DinoSamGrounding and returns a generic caption.
    This function replaces the previous Florence-based approach.

    Args:
        grounding_pipeline: An initialized DinoSamGrounding instance.
        image: The input PIL image.

    Returns:
        A tuple containing:
        - The generated mask as a numpy array.
        - A generic caption string for training the diffusion model.
    """
    mask = grounding_pipeline.generate_mask_from_text(
        image=image,
        text_prompt=FURNITURE_PROMPT,
        box_threshold=0.3,
        text_threshold=0.25,
        debug=debug,
    )
    return mask, GENERIC_CAPTION