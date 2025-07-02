MODELS_DIR="/kaggle/working/models"

# Step 0: Ensure a clean state for the model output directory before training begins.
echo "Cleaning up previous model directory: $MODELS_DIR"
rm -rf $MODELS_DIR

# Step 1: Prepare the data 
torchrun --nproc_per_node=2 prepare_data.py \
    --paired_dir "/kaggle/input/roomsdataset/500_empty_staged_384px" \
    --unpaired_dir "/kaggle/input/roomsdataset/unpaired_384px" \
    --output_dir "/kaggle/working/processed_data"

# Step 2: Run the two-phase training 
torchrun --nproc_per_node=2 train.py \
    --data_dir "/kaggle/working/processed_data" \
    --output_dir $MODELS_DIR \
    --batch_size 1 \
    --pretrain_epochs 1 \
    --finetune_epochs 2

# Step 3: Run inference
python inference.py \
    --controlnet_path "$MODELS_DIR/final_controlnet" \
    --input_image "/kaggle/input/roomsdataset/500_empty_staged_384px/398_empty.png" \
    --output_dir "/kaggle/working/inference_output" \
    --prompt "modern interior styling, Scandinavian minimalism, add detailed furniture, rugs, indoor plants, wall art, keep lighting, layout, floor, windows, ceiling, walls unchanged, photorealistic materials, soft textures, warm tones" \
    --negative_prompt "blurry, low quality, cartoon, anime, unrealistic, deformed, ugly, watermark, text, signature" \
    --seed 42