import os
import cv2
import json
import random
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import glob
import shutil
import torch

from data_preparation_tools import DataPreparationModels
from constants import STAGED, EMPTY, AGNOSTIC, MASK, CAPTION, PNG

def get_masks_batched(image_pils, models):
    """Generates masks for a batch of images."""
    with torch.no_grad():
        inputs = models.segmentation_processor(images=image_pils, return_tensors="pt").to(models.device)
        outputs = models.segmentation_model(**inputs)
        
    target_sizes = [img.size[::-1] for img in image_pils]
    results = models.segmentation_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    
    furniture_ids = [8, 13, 16, 18, 22, 28, 30, 32, 53] # ADE20K IDs for furniture
    batch_masks = []
    for semantic_map in results:
        mask = torch.zeros_like(semantic_map, dtype=torch.uint8)
        for furniture_id in furniture_ids:
            mask[semantic_map == furniture_id] = 255
        
        final_mask_np = mask.cpu().numpy()
        dilated_mask = cv2.dilate(final_mask_np, np.ones((5,5),np.uint8), iterations=3)
        batch_masks.append(dilated_mask)
        
    return batch_masks

def main(args):
    # --- Initialize Models ---
    models = DataPreparationModels(device='cuda:0')
    
    # --- Find all files ---
    paired_files = glob.glob(os.path.join(args.paired_dir, f"*_{STAGED}.png"))
    unpaired_files = glob.glob(os.path.join(args.unpaired_dir, "*.png")) if args.unpaired_dir else []
    
    print(f"Found {len(paired_files)} paired and {len(unpaired_files)} unpaired images.")

    # --- Process all files ---
    # We create a combined list to process all images at once to maximize GPU utilization
    all_files_to_process = (
        [{'path': p, 'type': 'paired'} for p in paired_files] +
        [{'path': p, 'type': 'unpaired'} for p in unpaired_files]
    )

    batch_size = 16 # Process 16 images at a time
    datalist_paired = []
    datalist_unpaired = []
    
    for i in tqdm(range(0, len(all_files_to_process), batch_size), desc="Processing All Images"):
        batch_info = all_files_to_process[i:i+batch_size]
        if not batch_info: continue

        batch_paths = [info['path'] for info in batch_info]
        batch_types = [info['type'] for info in batch_info]
        
        try:
            batch_images_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # 1. Get captions for the batch (this is fast)
            caption_results = models.captioner(batch_images_pil, max_new_tokens=50, batch_size=len(batch_images_pil))
            captions = [res[0]['generated_text'] for res in caption_results]
            
            # 2. Get masks for the batch
            masks = get_masks_batched(batch_images_pil, models)

            # 3. Save results
            for j, info in enumerate(batch_info):
                staged_path = info['path']
                data_type = info['type']
                mask = masks[j]
                caption = captions[j]
                
                staged_img = cv2.imread(staged_path)
                
                if data_type == 'paired':
                    pair_ix = os.path.basename(staged_path).split('_')[0]
                    empty_path = os.path.join(args.paired_dir, f"{pair_ix}_{EMPTY}.png")
                    agnostic_base_img = cv2.imread(empty_path)
                    output_dir = os.path.join(args.output_dir, "paired_processed")
                else: # unpaired
                    agnostic_base_img = np.copy(staged_img)
                    output_dir = os.path.join(args.output_dir, "unpaired_processed")
                
                os.makedirs(output_dir, exist_ok=True)
                
                agnostic = np.copy(agnostic_base_img)
                agnostic[mask == 255] = [127, 127, 127]
                
                base_name = os.path.basename(staged_path)
                staged_out = os.path.join(output_dir, base_name)
                agnostic_out = os.path.join(output_dir, base_name.replace('.png', f'_{AGNOSTIC}.png'))
                mask_out = os.path.join(output_dir, base_name.replace('.png', f'_{MASK}.png'))
                
                cv2.imwrite(staged_out, staged_img)
                cv2.imwrite(agnostic_out, agnostic)
                cv2.imwrite(mask_out, mask)
                
                item = {STAGED: staged_out, AGNOSTIC: agnostic_out, MASK: mask_out, CAPTION: caption}
                if data_type == 'paired':
                    empty_out = os.path.join(output_dir, os.path.basename(empty_path))
                    cv2.imwrite(empty_out, agnostic_base_img)
                    item[EMPTY] = empty_out
                    datalist_paired.append(item)
                else:
                    datalist_unpaired.append(item)

        except Exception as e:
            print(f"CRITICAL ERROR processing batch starting with {batch_paths[0]}: {e}. Skipping batch.")
            continue

    # --- Save final JSON files ---
    if datalist_paired:
        random.shuffle(datalist_paired)
        eval_list, train_list = datalist_paired[:50], datalist_paired[50:]
        with open(os.path.join(args.output_dir, 'paired_eval_datalist.json'), 'w') as f: json.dump(eval_list, f, indent=4)
        with open(os.path.join(args.output_dir, 'paired_train_datalist.json'), 'w') as f: json.dump(train_list, f, indent=4)
        print(f"\nSuccessfully processed and wrote {len(datalist_paired)} paired items.")
    
    if datalist_unpaired:
        with open(os.path.join(args.output_dir, 'unpaired_train_datalist.json'), 'w') as f: json.dump(datalist_unpaired, f, indent=4)
        print(f"Successfully processed and wrote {len(datalist_unpaired)} unpaired items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare virtual staging data.")
    parser.add_argument("--paired_dir", type=str, required=True, help="Path to the directory with paired images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where all processed data will be saved.")
    parser.add_argument("--unpaired_dir", type=str, default=None, help="(Optional) Path to the directory with unpaired images.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    main(args)