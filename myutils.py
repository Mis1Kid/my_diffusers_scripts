from diffusers.models.embeddings import ImageProjection
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import random
import argparse


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def create_ip_adapter_state_dict(unet):
    # "ip_adapter" (cross-attention weights)
    ip_cross_attn_state_dict = {}
    key_id = 1
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") or "motion_module" in name else unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            layer_name = name.split(".processor")[0]
            ip_cross_attn_state_dict.update(
                {
                    f"{key_id}.to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    f"{key_id}.to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
            )

            key_id += 2

    # "image_proj" (ImageProjection layer weights)
    cross_attention_dim = unet.config["cross_attention_dim"]
    image_projection = ImageProjection(
        cross_attention_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, num_image_text_embeds=4
    )
    torch.nn.init.zeros_(image_projection.image_embeds.weight)
    torch.nn.init.zeros_(image_projection.image_embeds.bias)
    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


def convert_unet_ip_adapter(unet):
    ip_adapter_sd = {}
    image_proj = {}
    ip_adapter_attn = {}

    for key, value in unet.encoder_hid_proj.state_dict().items():
        diffusers_name = key.replace("image_projection_layers.0.", "").replace("image_embeds", "proj")
        image_proj[diffusers_name] = value

    unet_sd = unet.state_dict()
    key_id = 1
    for i, name in enumerate(unet.attn_processors.keys()):
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is not None:
            layer_name = name.split(".processor")[0]
            ip_adapter_attn.update(
                {
                    f"{key_id}.to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    f"{key_id}.to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
            )
            key_id += 2
    ip_adapter_sd = {"image_proj": image_proj, "ip_adapter": ip_adapter_attn}
    return ip_adapter_sd


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=r"runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None)
    parser.add_argument(
        "--pretrained_dinov2_name_or_path",
        type=str,
        default=r"facebook/dinov2-base",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--data_json_file", type=str, default=None)
    parser.add_argument("--data_root_path", type=str, default=r"/mnt/d/workspace/Datasets/TextureInpaint")
    parser.add_argument("--output_dir", type=str, default=r"/mnt/d/workspace/Output/diffusers-lora-v2")
    parser.add_argument("--instance_prompt", type=str, default="a picture of liuyin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=2000)
    parser.add_argument("--max_train_steps", type=int, default=4000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--validation_epochs", type=int, default=200)
    parser.add_argument("--validation_prompt", type=str, default="liuyin")
    parser.add_argument("--num_validation_images", type=int, default=1)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
