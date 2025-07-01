import torch
from transformers import pipeline, AutoImageProcessor, MaskFormerForInstanceSegmentation

class DataPreparationModels:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        print(f"Loading models onto {self.device}...")
        
        # Load captioning pipeline
        self.captioner = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-large", 
            device=self.device
        )
        
        # Load segmentation model
        self.segmentation_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade")
        self.segmentation_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade").to(self.device)
        
        print("All models loaded.")