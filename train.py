import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import gc

from dataset import VirtualStagingDataset
import random

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across various libraries and CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

class ControlNetTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning module for training ControlNet with Stable Diffusion.
    """
    def __init__(self, model_name: str, controlnet_path: str, lr: float):
        super().__init__()
        self.save_hyperparameters()

        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled.")
        except Exception:
            print("xformers not available. Running without memory efficient attention.")

        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.controlnet = self.pipeline.controlnet
        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.train()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Perform a single training step.
        """
        latents = self.vae.encode(batch["pixel_values"].to(self.device, dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.text_encoder(
            self.tokenizer(batch["prompt"], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(self.device)
        )[0]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=batch["conditioning_pixel_values"].to(self.device, dtype=self.controlnet.dtype),
            return_dict=False,
        )
        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for training.
        """
        optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=self.hparams.lr)
        return optimizer

def run_pretraining(args, model_name: str, base_controlnet_path: str) -> str:
    """
    Run the pre-training phase on unpaired data if available.

    Returns:
        str: Path to the pre-trained ControlNet model.
    """
    unpaired_datalist_path = os.path.join(args.data_dir, "unpaired_train_datalist.json")
    if not os.path.exists(unpaired_datalist_path):
        print("Unpaired datalist not found. Skipping pre-training phase.")
        return base_controlnet_path

    print("\n" + "="*50)
    print("PHASE 1: UNPAIRED PRE-TRAINING FOR REALISM")
    print("="*50)

    unpaired_dataset = VirtualStagingDataset(data_list_path=unpaired_datalist_path)
    unpaired_dataloader = DataLoader(unpaired_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    pretrain_model = ControlNetTrainingModule(model_name, base_controlnet_path, lr=args.pretrain_lr)

    pretrain_trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp_find_unused_parameters_true",
        precision="16-mixed", max_epochs=args.pretrain_epochs,
        default_root_dir=os.path.join(args.output_dir, "pretrain_logs"),
        enable_checkpointing=False,
        logger=False,
    )
    pretrain_trainer.fit(pretrain_model, unpaired_dataloader)

    pre_trained_controlnet_path = os.path.join(args.output_dir, "pretrained_controlnet")
    if pretrain_trainer.is_global_zero:
        print("Rank 0 saving pre-trained model...")
        pretrain_model.controlnet.save_pretrained(pre_trained_controlnet_path)

    pretrain_trainer.strategy.barrier()

    del pretrain_model, unpaired_dataloader, pretrain_trainer
    gc.collect(); torch.cuda.empty_cache()
    print("Pre-training complete. Memory flushed.")

    return pre_trained_controlnet_path

def run_finetuning(args, model_name: str, controlnet_path: str) -> None:
    """
    Run the fine-tuning phase on paired data.

    Returns:
        None
    """
    print("\n" + "="*50)
    print("PHASE 2: PAIRED FINE-TUNING FOR STRUCTURE PRESERVATION")
    print("="*50)

    paired_datalist_path = os.path.join(args.data_dir, "paired_train_datalist.json")
    paired_dataset = VirtualStagingDataset(data_list_path=paired_datalist_path)
    paired_dataloader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    finetune_model = ControlNetTrainingModule(model_name, controlnet_path, lr=args.finetune_lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "final_checkpoints"),
        filename="best-model-{epoch}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    finetune_trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp_find_unused_parameters_true",
        precision="16-mixed", max_epochs=args.finetune_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=os.path.join(args.output_dir, "finetune_logs"),
    )
    finetune_trainer.fit(finetune_model, paired_dataloader)

    final_controlnet_path = os.path.join(args.output_dir, "final_controlnet")
    if finetune_trainer.is_global_zero:
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            try:
                best_model = ControlNetTrainingModule.load_from_checkpoint(best_model_path)
                best_model.controlnet.save_pretrained(final_controlnet_path)
                print(f"Fine-tuning complete. Final best model saved to {final_controlnet_path}")
            except Exception as e:
                print(f"Could not load best checkpoint, error: {e}. Saving last model instead.")
                finetune_model.controlnet.save_pretrained(final_controlnet_path)
        else:
            print("No best model checkpoint found. Saving last model instead.")
            finetune_model.controlnet.save_pretrained(final_controlnet_path)

def main(args) -> None:
    """
    Main function to run pre-training and fine-tuning phases.
    """
    model_name = "runwayml/stable-diffusion-v1-5"
    base_controlnet_path = "lllyasviel/control_v11p_sd15_inpaint"

    phase2_controlnet_path = run_pretraining(args, model_name, base_controlnet_path)
    run_finetuning(args, model_name, phase2_controlnet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the virtual staging ControlNet model.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--pretrain_lr", type=float, default=5e-6)
    parser.add_argument("--finetune_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    main(args)