# train_lora.py (Corrected Version)

import os
import argparse
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

def main(args):
    #  Configuration 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #  Load the Dataset
    dataset_path = 'dataset'
    csv_path = os.path.join(dataset_path, 'metadata.csv')
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['image_path'] = df['file_name'].apply(lambda x: os.path.join(dataset_path, 'images', x))
    hf_dataset = Dataset.from_pandas(df)
    print(f"Dataset loaded with {len(hf_dataset)} entries.")

    #  Preprocess the Dataset 
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    
    # Pre-compute image tensors and token IDs to avoid repeated processing
    def preprocess_function(examples):
        images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
        pixel_values = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1) / 127.5 - 1.0 for img in images
        ]).to(torch.float32)

        input_ids = tokenizer(
            examples["text"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    processed_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=hf_dataset.column_names)
    processed_dataset.set_format(type="torch")

    train_dataloader = torch.utils.data.DataLoader(
        processed_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    print("Dataset preprocessed and DataLoader created.")

    #  Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    
    # Freeze original model parameters
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    print("Base models loaded and frozen.")

    #  Configure and Apply LoRA 
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_unet = get_peft_model(unet, lora_config)
    print("LoRA configured and applied to U-Net.")

    #  Set up Training 
    optimizer = torch.optim.AdamW(lora_unet.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision == 'fp16'))

    #  Training Loop
    print(" Starting Training ")
    progress_bar = tqdm(range(args.num_train_epochs * len(train_dataloader)))
    progress_bar.set_description("Training Steps")

    for epoch in range(args.num_train_epochs):
        lora_unet.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.amp.autocast('cuda', enabled=(args.mixed_precision == 'fp16')):
                # Get text embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict the noise residual
                model_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate the loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())

    print(" Training Finished ")

    #  Save the Trained LoRA Weights 
    lora_unet.save_pretrained(args.output_dir)
    print(f"LoRA model weights saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-2-base", help="Path to the base model.")
    parser.add_argument("--output_dir", type=str, default="lora-floorplan-model", help="Directory to save the trained LoRA weights.")
    parser.add_argument("--lora_rank", type=int, default=16, help="The rank of the LoRA matrices.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"], help="Use mixed precision training.")
    
    args = parser.parse_args()
    main(args)