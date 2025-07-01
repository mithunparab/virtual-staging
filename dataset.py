import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VirtualStagingDataset(Dataset):
    def __init__(self, data_list_path, image_size=512):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        target_image = Image.open(item["staged"]).convert("RGB")
        # The agnostic image becomes our primary condition.
        conditioning_image = Image.open(item["agnostic"]).convert("RGB")
        prompt = item["caption"]
        
        target_image = self.image_transforms(target_image)
        conditioning_image = self.conditioning_image_transforms(conditioning_image)

        # Normalize target image to [-1, 1] for VAE
        target_image = (target_image * 2.0) - 1.0
        
        return {
            "pixel_values": target_image,
            "conditioning_pixel_values": conditioning_image,
            "prompt": prompt,
        }