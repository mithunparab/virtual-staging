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
import traceback
import torch
import torch.distributed as dist

from layout_generator import DinoSamGrounding, get_grounded_mask
from constants import STAGED, EMPTY, AGNOSTIC, MASK, CAPTION

def main(args, rank: int, world_size: int, device: str) -> None:
    """
    Prepares data for virtual staging by generating masks, captions, and agnostic images
    using GroundingDINO and SAM models. Handles both paired and unpaired images, supports
    distributed processing, and saves processed data and metadata to output directory.
    This version replaces the Florence-based approach with GroundingDINO to create
    more robust and complete masks for furniture.

    Args:
        args: Parsed command-line arguments.
        rank: Process rank for distributed processing.
        world_size: Total number of processes.
        device: Device identifier (e.g., 'cuda:0').
    Returns:
        None
    """
    is_main_process = (rank == 0)

    try:
        grounding_pipeline = DinoSamGrounding(device=device)
        grounding_pipeline.load(
            config_path=args.dino_config,
            checkpoint_path=args.dino_checkpoint
        )
    except Exception as e:
        print(f"GPU {rank} FATAL: Could not initialize or load GroundingDINO/SAM models. Error: {e}")
        print("Please ensure GroundingDINO is installed and model weights are available.")
        traceback.print_exc()
        if world_size > 1: dist.barrier()
        return

    all_files_to_process = []
    if is_main_process:
        paired_files = glob.glob(os.path.join(args.paired_dir, f"*_{STAGED}.png"))
        unpaired_files = glob.glob(os.path.join(args.unpaired_dir, "*.png")) if args.unpaired_dir else []
        print(f"Found {len(paired_files)} paired and {len(unpaired_files)} unpaired images.")
        all_files_to_process = ([{'path': p, 'type': 'paired'} for p in paired_files] +
                              [{'path': p, 'type': 'unpaired'} for p in unpaired_files])
        random.shuffle(all_files_to_process)

    if world_size > 1:
        file_list_obj = [all_files_to_process]
        dist.broadcast_object_list(file_list_obj, src=0)
        all_files_to_process = file_list_obj[0]

    files_for_this_rank = all_files_to_process[rank::world_size]
    
    processed_items = []
    progress_bar = tqdm(files_for_this_rank, desc=f"GPU {rank} Processing", position=rank, dynamic_ncols=True)
    
    for info in progress_bar:
        staged_path = info['path']
        try:
            staged_img_pil = Image.open(staged_path).convert("RGB")
            
            mask, caption = get_grounded_mask(grounding_pipeline, staged_img_pil, False)

            if np.sum(mask) == 0:
                print(f"Warning: No mask generated for {os.path.basename(staged_path)}. Skipping file.")
                continue
            
            staged_img_cv = cv2.imread(staged_path)
            data_type = info['type']
            
            if data_type == 'paired':
                pair_ix = os.path.basename(staged_path).split('_')[0]
                empty_path = os.path.join(args.paired_dir, f"{pair_ix}_{EMPTY}.png")
                agnostic_base_img = cv2.imread(empty_path)
                output_dir = os.path.join(args.output_dir, "paired_processed")
            else:
                agnostic_base_img = np.copy(staged_img_cv)
                output_dir = os.path.join(args.output_dir, "unpaired_processed")
            
            os.makedirs(output_dir, exist_ok=True)
            agnostic = np.copy(agnostic_base_img); agnostic[mask == 255] = [127, 127, 127]
            base_name = os.path.basename(staged_path)
            
            staged_out = os.path.join(output_dir, base_name)
            agnostic_out = os.path.join(output_dir, base_name.replace('.png', f'_{AGNOSTIC}.png'))
            mask_out = os.path.join(output_dir, base_name.replace('.png', f'_{MASK}.png'))
            
            cv2.imwrite(staged_out, staged_img_cv)
            cv2.imwrite(agnostic_out, agnostic)
            cv2.imwrite(mask_out, mask)
            
            item = {STAGED: staged_out, AGNOSTIC: agnostic_out, MASK: mask_out, CAPTION: caption, 'type': data_type}
            if data_type == 'paired':
                empty_out = os.path.join(output_dir, os.path.basename(empty_path))
                cv2.imwrite(empty_out, agnostic_base_img)
                item[EMPTY] = empty_out
            processed_items.append(item)

        except Exception as e:
            print(f"GPU {rank} CRITICAL ERROR on {os.path.basename(staged_path)}: {e}")
            traceback.print_exc()
            continue
    
    grounding_pipeline.release()

    all_processed_items = [None] * world_size if world_size > 1 else [processed_items]
    if world_size > 1:
        dist.barrier()
        dist.all_gather_object(all_processed_items, processed_items)

    if is_main_process:
        print("\nMain process gathering and finalizing results...")
        final_paired, final_unpaired = [], []
        all_items_flat = [item for sublist in all_processed_items if sublist is not None for item in sublist]

        for item in all_items_flat:
            if item['type'] == 'paired':
                final_paired.append(item)
            else:
                final_unpaired.append(item)
        
        if final_paired:
            eval_count = min(50, len(final_paired) // 10) if len(final_paired) > 50 else 0
            eval_list, train_list = final_paired[:eval_count], final_paired[eval_count:]
            if eval_list:
                with open(os.path.join(args.output_dir, 'paired_eval_datalist.json'), 'w') as f:
                    json.dump(eval_list, f, indent=4)
            if train_list:
                with open(os.path.join(args.output_dir, 'paired_train_datalist.json'), 'w') as f:
                    json.dump(train_list, f, indent=4)
            print(f"\nSuccessfully processed {len(final_paired)} paired items.")
        if final_unpaired:
            with open(os.path.join(args.output_dir, 'unpaired_train_datalist.json'), 'w') as f:
                json.dump(final_unpaired, f, indent=4)
            print(f"Successfully processed {len(final_unpaired)} unpaired items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data with the GroundingDINO+SAM pipeline.")
    parser.add_argument("--paired_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unpaired_dir", type=str, default=None)
    parser.add_argument("--dino_config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to GroundingDINO config file.")
    parser.add_argument("--dino_checkpoint", type=str, default="groundingdino_swint_ogc.pth", help="Path to GroundingDINO checkpoint file.")
    args = parser.parse_args()

    rank, world_size, device = 0, 1, 'cuda:0'
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'
        print(f"Initialized process with rank {rank} on device {device}.")

    if rank == 0:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    if is_distributed:
        dist.barrier()
    
    try:
        main(args, rank, world_size, device)
    finally:
        if is_distributed:
            dist.destroy_process_group()