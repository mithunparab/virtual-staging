import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VirtualStagingDataset(Dataset):
    """
    PyTorch Dataset for virtual staging tasks.

    Args:
        data_list_path (str): Path to the JSON file containing dataset entries.
        image_size (int, optional): Size to which images are resized and cropped. Default is 512.

    Each entry in the JSON file should be a dict with keys:
        - "staged": Path to the target (staged) image.
        - "agnostic": Path to the conditioning (agnostic) image.
        - "caption": Text prompt describing the image.

    Returns:
        dict: {
            "pixel_values": torch.Tensor,               # Target image, normalized to [-1, 1]
            "conditioning_pixel_values": torch.Tensor,  # Conditioning image, normalized to [0, 1]
            "prompt": str                               # Caption text
        }
    """
    def __init__(self, data_list_path: str, image_size: int = 512):
        with open(data_list_path, 'r') as f:
            self.data = json.load(f)
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        target_image = Image.open(item["staged"]).convert("RGB")
        conditioning_image = Image.open(item["agnostic"]).convert("RGB")
        prompt = item["caption"]
        
        target_image = self.image_transforms(target_image)
        conditioning_image = self.conditioning_image_transforms(conditioning_image)
        target_image = (target_image * 2.0) - 1.0
        
        return {
            "pixel_values": target_image,
            "conditioning_pixel_values": conditioning_image,
            "prompt": prompt,
        }