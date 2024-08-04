import argparse
import itertools
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoImageProcessor, Dinov2Model

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from myutils import convert_unet_ip_adapter, parse_args, create_ip_adapter_state_dict
from texture_dataset import MyDataset
from accelerate import accelerator


def main():
    args = parse_args()
    args.output_dir = str(Path(args.output_dir).parent / "train_dinov2_adapter")
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    feature_extractor = AutoImageProcessor.from_pretrained(args.pretrained_dinov2_name_or_path)
    image_encoder = Dinov2Model.from_pretrained(args.pretrained_dinov2_name_or_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)

    ip_adapter_sd = create_ip_adapter_state_dict(unet)
    unet._load_ip_adapter_weights(ip_adapter_sd)
    ip_adapter_params = filter(lambda p: p.requires_grad, unet.parameters())
    ip_adapter_params = list(ip_adapter_params)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        ip_adapter_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = MyDataset(args.data_root_path, tokenizer=tokenizer, feature_extractor=feature_extractor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("train_dinov2_adapter", config=vars(args))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process
    )
    global_step = 0
    first_epoch = 0
    train_loss = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # if accelerator.is_main_process and step == 0:
            #     print("epoch: ", epoch)
            with accelerator.accumulate(unet):
                # Convert images to latent space

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_model_input = noisy_latents

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the image embedding for conditioning
                image_embedding = image_encoder(
                    batch["image_features"].to(accelerator.device, dtype=weight_dtype)
                ).pooler_output.unsqueeze(1)
                added_cond_kwargs = {"image_embeds": [image_embedding]}

                # Predict the noise residual
                noise_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                train_loss += loss.detach().cpu().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(filter(lambda p: p.requires_grad, unet.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_main_process:
                    if global_step % args.logging_steps == 0:
                        train_loss = train_loss / args.logging_steps
                        print(f"Step {global_step}, Loss: {train_loss} ,lr: {lr_scheduler.get_last_lr()[0]}")
                        train_loss = 0
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwarpped_unet = accelerator.unwrap_model(unet)
                        ip_adapter_sd_to_save = convert_unet_ip_adapter(unwarpped_unet)
                        torch.save(ip_adapter_sd_to_save, os.path.join(save_path, f"dinov2_encoder-{global_step}.pth"))
                        print(f"Saving checkpoint at step {global_step} to {save_path}")

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
