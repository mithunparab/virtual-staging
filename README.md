# Virtual Staging with Stable Diffusion

This project demonstrates how to perform virtual staging on empty rooms using Stable Diffusion and ControlNet. The goal is to generate realistic staged photos while preserving the structural integrity of the original room.

## Approach

The pipeline uses a two-phase training strategy:

1. **Unpaired Pre-training:** Fine-tunes a ControlNet model on a dataset of staged images to learn realistic furniture styles and compositions, independent of specific room layouts.
2. **Paired Fine-tuning:** Further fine-tunes the model on a dataset of empty rooms paired with their staged counterparts. This phase teaches the model to preserve the original room's structure (walls, windows, lighting) while in-painting the furniture learned in the first phase, guided by text prompts.

The data preparation uses stable and reliable models (BLIP for captions, MaskFormer for segmentation) to create the necessary training data.

## Setup

Ensure you have the necessary libraries installed. The `requirements.txt` file in this repository lists all dependencies.

## Workflow

The process is divided into three main stages: Data Preparation, Training, and Inference.

### 1. Data Preparation

This stage processes the raw image data into formats suitable for training. It involves:

* Generating captions for staged images.
* Segmenting furniture from staged images to create layout masks.
* Creating "agnostic" images by masking furniture in empty rooms.
* Splitting data into training and validation sets.

**Commands:**

1. **Set environment variables (replace paths accordingly):**

    ```bash
    export PAIRED_DATA_DIR="/kaggle/input/roomsdataset/500_empty_staged_384px"
    export UNPAIRED_DATA_DIR="/kaggle/input/roomsdataset/unpaired_384px"
    export PROCESSED_DATA_DIR="/kaggle/working/processed_data"
    ```

2. **Create output directory and clean up previous runs:**

    ```bash
    rm -rf $PROCESSED_DATA_DIR
    mkdir -p $PROCESSED_DATA_DIR
    ```

3. **Run the data preparation script:**

    ```bash
    python prepare_data.py --paired_dir $PAIRED_DATA_DIR --output_dir $PROCESSED_DATA_DIR --unpaired_dir $UNPAIRED_DATA_DIR
    ```

    This script will create `paired_train_datalist.json`, `paired_eval_datalist.json`, and `unpaired_train_datalist.json` in your output directory.

### 2. Training

This stage fine-tunes the ControlNet model. It consists of two phases: unpaired pre-training and paired fine-tuning.

**Commands:**

1. **Ensure `requirements.txt` is up-to-date:**
    * Make sure `torch==2.3.0` and `xformers==0.0.26.post1` are installed.
    * Run: `pip install -r requirements.txt`

2. **Run the training script (distributed across 2 GPUs):**

    ```bash
    torchrun --nproc_per_node=2 train.py \
        --data_dir $PROCESSED_DATA_DIR \
        --output_dir "/kaggle/working/models" \
        --batch_size 1 \
        --pretrain_epochs 15 \
        --finetune_epochs 10 \
        --pretrain_lr 5e-6 \
        --finetune_lr 1e-6
    ```

    This command will perform the two-phase training and save the final model to `/kaggle/working/models/final_controlnet`.

### 3. Inference

Once the model is trained, you can use it to stage new, empty room images.

**Commands:**

1. **Run the inference script:**

    ```bash
    python inference.py \
        --controlnet_path "/kaggle/working/models/final_controlnet" \
        --input_image "/path/to/your/empty_room_image.png" \
        --output_dir "/kaggle/working/inference_output" \
        --prompt "A modern, minimalist living room with a white sofa, a glass coffee table, and a tall green plant. Natural light from a large window." \
        --seed 1234
    ```

    * Replace `/path/to/your/empty_room_image.png` with the actual path to your input image.
    * Adjust the `--prompt` to describe your desired staging.
    * The output images will be saved in `/kaggle/working/inference_output`.

## Limitations & Outlook

This project serves as a proof-of-concept. Areas for improvement include:

* [ ] More sophisticated prompt engineering for style control  
* [ ] Handling existing furniture in input images  
* [ ] Finer control over object placement and style  
* [ ] More robust error handling and logging  
* [ ] Potentially exploring faster or more memory-efficient model architectures  
