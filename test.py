# %%
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import Dinov2Model, AutoImageProcessor
from diffusers.models.modeling_utils import load_state_dict
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

pipeline_path = r"runwayml/stable-diffusion-v1-5"
dinov2_path = r"facebook/dinov2-base"

# print(torch.__version__)
# print(torch.cuda.is_available())
pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(pipeline_path)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
pipeline.safety_checker = None
dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_path)
dinov2_model = Dinov2Model.from_pretrained(dinov2_path).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(123456789)


# %%
def get_embeddings(processor, model, images, with_negatives=False):
    features = processor(images, return_tensors="pt").to("cuda")
    outputs = model(**features)
    if "pooler_output" in outputs.keys():
        embeddings = outputs.pooler_output
    if "image_embeds" in outputs.keys():
        embeddings = outputs.image_embeds
    if embeddings.ndim < 3:
        embeddings = embeddings.unsqueeze(1)
    if with_negatives:
        neg_embeddings = torch.zeros_like(embeddings)
        embeddings = torch.cat([embeddings, neg_embeddings], dim=0)
    return embeddings


def generate(
    pipeline, prompts, lora_path=None, ip_adapter_path=None, processor=None, image_encoder=None, ref_images=None
):
    generator = torch.Generator(device="cuda").manual_seed(123456789)
    pipeline.unload_ip_adapter()
    pipeline.unload_lora_weights()
    outputs_raw = pipeline(prompt=prompts, generator=generator).images

    if lora_path is not None:
        pipeline.load_lora_weights(lora_path)
    if ip_adapter_path is not None:
        ip_adapter_sd = torch.load(ip_adapter_path)
        pipeline.unet._load_ip_adapter_weights(ip_adapter_sd)
    if processor is not None and image_encoder is not None and ref_images is not None:
        embeddings = get_embeddings(processor, image_encoder, ref_images, with_negatives=True)
    else:
        embeddings = None
    outputs = pipeline(
        prompt=prompts, guidance_scale=0.0, ip_adapter_image_embeds=[embeddings], generator=generator
    ).images

    return outputs_raw + outputs


prompts = ["liuyin"]
lora_path = r"/mnt/d/workspace/Output/diffusers-lora-v2/lora_weights.pth"
ip_adapter_path = r"/mnt/d/workspace/Output/train_dinov2_adapter/checkpoint-4000/dinov2_encoder-4000.pth"
ref_path = r"/home/hcl/workspace/my_scripts/assert/liuyin/liuyin-1.png"

ref_images = [Image.open(ref_path).convert("RGB")]

outputs = generate(pipeline, prompts, None, ip_adapter_path, dinov2_processor, dinov2_model, ref_images=ref_images)
for i, output in enumerate(outputs):
    output.save(f"output_{i}.png")
# %%
