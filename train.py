import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import random

from dataset import VirtualStagingDataset

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across various libraries and CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ControlNetTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning module for training ControlNet with Stable Diffusion and a pre-loaded LoRA.
    """
    def __init__(self, model_name: str, controlnet_path: str, lr: float, 
                 lora_repo: str, lora_weights: str, lora_subfolder: str = None):
        super().__init__()
        self.save_hyperparameters()

        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        print(f"Attempting to load LoRA from repo: {self.hparams.lora_repo}")
        lora_kwargs = {"weight_name": self.hparams.lora_weights}
        if self.hparams.lora_subfolder:
            lora_kwargs["subfolder"] = self.hparams.lora_subfolder
            print(f"Using subfolder: {self.hparams.lora_subfolder}")
        
        try:
            self.pipeline.load_lora_weights(self.hparams.lora_repo, **lora_kwargs)
            print("✅ Successfully loaded LoRA weights.")
        except Exception as e:
            print(f"🛑 Failed to load LoRA weights. Error: {e}")
            raise e

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

def run_training(args, model_name: str, controlnet_path: str) -> None:
    """
    Run the training phase on paired data.
    """
    print("\n" + "="*50)
    print("STARTING CONTROLNET TRAINING WITH PRE-LOADED LORA")
    print("="*50)

    paired_datalist_path = os.path.join(args.data_dir, "paired_train_datalist.json")
    paired_dataset = VirtualStagingDataset(data_list_path=paired_datalist_path)
    paired_dataloader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    training_model = ControlNetTrainingModule(
        model_name, controlnet_path, 
        lr=args.learning_rate,
        lora_repo=args.lora_model_repo,
        lora_weights=args.lora_weight_name,
        lora_subfolder=args.lora_subfolder
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "final_checkpoints"),
        filename="best-model-{epoch}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp_find_unused_parameters_true",
        precision="16-mixed", max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=os.path.join(args.output_dir, "training_logs"),
    )
    trainer.fit(training_model, paired_dataloader)

    final_controlnet_path = os.path.join(args.output_dir, "final_controlnet")
    if trainer.is_global_zero:
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            try:
                best_model = ControlNetTrainingModule.load_from_checkpoint(best_model_path)
                best_model.controlnet.save_pretrained(final_controlnet_path)
                print(f"Training complete. Final best model saved to {final_controlnet_path}")
            except Exception as e:
                print(f"Could not load best checkpoint, error: {e}. Saving last model instead.")
                training_model.controlnet.save_pretrained(final_controlnet_path)
        else:
            print("No best model checkpoint found. Saving last model instead.")
            training_model.controlnet.save_pretrained(final_controlnet_path)

def main(args) -> None:
    """
    Main function to run the training.
    """
    if args.seed is not None:
        set_seed(args.seed)
    
    run_training(args, args.model_name, args.controlnet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the virtual staging ControlNet model with a LoRA.")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training data list.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save logs and model checkpoints.")
    
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model.")
    parser.add_argument("--controlnet_path", type=str, default="lllyasviel/control_v11p_sd15_inpaint", help="Base ControlNet model.")
    
    parser.add_argument("--lora_model_repo", type=str, required=True, help="Hugging Face repo ID of the LoRA model.")
    parser.add_argument("--lora_weight_name", type=str, default="pytorch_lora_weights.safetensors", help="Filename of the LoRA weights within the repo.")
    parser.add_argument("--lora_subfolder", type=str, default=None, help="Optional: subfolder within the repo if loading from a checkpoint (e.g., 'checkpoint-3000').")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)