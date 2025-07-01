import os
import cv2
import json
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

from layout_generator import FlorenceLayoutGenerator
from constants import (
    STAGED, AGNOSTIC, MASK, CAPTION, PNG
)

# Read from environment variables
UNPAIRED_DATA_DIR_INPUT = os.getenv("UNPAIRED_DATA_DIR", "/kaggle/input/roomsdataset/unpaired_384px")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/kaggle/working/processed_data")

def prepare_unpaired_data():
    """
    Prepares the unpaired dataset for pre-training.
    It creates 'pseudo-agnostic' images from the staged images.
    """
    output_dir = os.path.join(PROCESSED_DATA_DIR, "unpaired_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Using glob to find all png images in the directory
    image_paths = glob.glob(os.path.join(UNPAIRED_DATA_DIR_INPUT, "*.png"))
    print(f"Found {len(image_paths)} unpaired images to process.")

    if not image_paths:
        print("ERROR: No unpaired images found. Check UNPAIRED_DATA_DIR_INPUT path.")
        return

    layout_gen = FlorenceLayoutGenerator(device='cuda:0')
    datalist = []

    for img_path in tqdm(image_paths, desc="Preparing Unpaired Data"):
        staged_img = cv2.imread(img_path)
        if staged_img is None:
            continue
        
        staged_img_rgb = cv2.cvtColor(staged_img, cv2.COLOR_BGR2RGB)
        staged_img_pil = Image.fromarray(staged_img_rgb)

        # Generate caption and mask
        caption = layout_gen.generate_captions_batch([staged_img_pil])[0]
        layout_mask = layout_gen.generate_layout_masks_batch([staged_img_pil], [caption])[0]

        # Create the "pseudo-agnostic" image by masking the staged image itself
        pseudo_agnostic = np.copy(staged_img)
        pseudo_agnostic[layout_mask == 255] = [127, 127, 127]

        # Define paths for the new data
        base_name = os.path.basename(img_path)
        agnostic_path = os.path.join(output_dir, base_name.replace(".png", f"_{AGNOSTIC}.png"))
        mask_path = os.path.join(output_dir, base_name.replace(".png", f"_{MASK}.png"))
        staged_path_output = os.path.join(output_dir, base_name)

        # Write data to disk
        cv2.imwrite(agnostic_path, pseudo_agnostic)
        cv2.imwrite(mask_path, layout_mask)
        cv2.imwrite(staged_path_output, staged_img) # Copy original for easy access

        datalist.append({
            AGNOSTIC: agnostic_path, # This is our control image
            STAGED: staged_path_output, # This is our target
            MASK: mask_path,
            CAPTION: caption
        })

    # Save the datalist for the unpaired set
    with open(os.path.join(PROCESSED_DATA_DIR, 'unpaired_train_datalist.json'), 'w') as fp:
        json.dump(datalist, fp, indent=4)

if __name__ == "__main__":
    prepare_unpaired_data()