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

from layout_generator import FlorenceModel, get_caption, get_mask_from_regions
from constants import STAGED, EMPTY, AGNOSTIC, MASK, CAPTION

def main(args):
    # --- Initialize Florence-2 Model ---
    florence_wrapper = FlorenceModel(device='cuda:0')
    florence_wrapper.load()
    
    # --- Find all files ---
    paired_files = glob.glob(os.path.join(args.paired_dir, f"*_{STAGED}.png"))
    unpaired_files = glob.glob(os.path.join(args.unpaired_dir, "*.png")) if args.unpaired_dir else []
    
    print(f"Found {len(paired_files)} paired and {len(unpaired_files)} unpaired images.")

    all_files_to_process = (
        [{'path': p, 'type': 'paired'} for p in paired_files] +
        [{'path': p, 'type': 'unpaired'} for p in unpaired_files]
    )

    datalist_paired = []
    datalist_unpaired = []
    
    for info in tqdm(all_files_to_process, desc="Processing Images with Florence-2"):
        staged_path = info['path']
        data_type = info['type']

        try:
            staged_img_pil = Image.open(staged_path).convert("RGB")
            
            # 1. Generate detailed caption using Florence-2
            caption = get_caption(florence_wrapper, staged_img_pil)
            
            # 2. Generate intelligent furniture mask using Florence-2
            mask = get_mask_from_regions(florence_wrapper, staged_img_pil)

            # Check for empty mask, which indicates a problem or an empty room image
            if np.sum(mask) == 0:
                print(f"Warning: No furniture mask generated for {os.path.basename(staged_path)}. Skipping file.")
                continue

            # 3. Save results
            staged_img_cv = cv2.imread(staged_path)
            
            if data_type == 'paired':
                pair_ix = os.path.basename(staged_path).split('_')[0]
                empty_path = os.path.join(args.paired_dir, f"{pair_ix}_{EMPTY}.png")
                agnostic_base_img = cv2.imread(empty_path)
                output_dir = os.path.join(args.output_dir, "paired_processed")
            else: # unpaired
                agnostic_base_img = np.copy(staged_img_cv)
                output_dir = os.path.join(args.output_dir, "unpaired_processed")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Create agnostic image by applying the mask
            agnostic = np.copy(agnostic_base_img)
            agnostic[mask == 255] = [127, 127, 127] # Gray mask
            
            base_name = os.path.basename(staged_path)
            staged_out = os.path.join(output_dir, base_name)
            agnostic_out = os.path.join(output_dir, base_name.replace('.png', f'_{AGNOSTIC}.png'))
            mask_out = os.path.join(output_dir, base_name.replace('.png', f'_{MASK}.png'))
            
            cv2.imwrite(staged_out, staged_img_cv)
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
            print(f"CRITICAL ERROR processing {staged_path}: {e}. Skipping file.")
            continue
    
    # --- Release Florence-2 from memory ---
    florence_wrapper.release()

    # --- Save final JSON files ---
    if datalist_paired:
        random.shuffle(datalist_paired)
        eval_count = min(50, len(datalist_paired) // 10) # 10% for eval, max 50
        eval_list, train_list = datalist_paired[:eval_count], datalist_paired[eval_count:]
        with open(os.path.join(args.output_dir, 'paired_eval_datalist.json'), 'w') as f: json.dump(eval_list, f, indent=4)
        with open(os.path.join(args.output_dir, 'paired_train_datalist.json'), 'w') as f: json.dump(train_list, f, indent=4)
        print(f"\nSuccessfully processed and wrote {len(datalist_paired)} paired items.")
    
    if datalist_unpaired:
        with open(os.path.join(args.output_dir, 'unpaired_train_datalist.json'), 'w') as f: json.dump(datalist_unpaired, f, indent=4)
        print(f"Successfully processed and wrote {len(datalist_unpaired)} unpaired items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare virtual staging data using Florence-2.")
    parser.add_argument("--paired_dir", type=str, required=True, help="Path to the directory with paired images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where all processed data will be saved.")
    parser.add_argument("--unpaired_dir", type=str, default=None, help="(Optional) Path to the directory with unpaired images.")
    args = parser.parse_args()

    # Clean up previous runs
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    
    main(args)