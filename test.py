# %%
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import (
    Dinov2Model,
    AutoImageProcessor,
    CLIPFeatureExtractor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTokenizer,
)
from diffusers.models.modeling_utils import load_state_dict
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
from myutils import load_ip_adapter_to_unet

HF_HOME = os.environ["HF_HOME"]
print(HF_HOME)
pipeline_path = r"runwayml/stable-diffusion-v1-5"
dinov2_path = r"facebook/dinov2-base"
clip_path = r"openai/clip-vit-large-patch14"
tokenizer_path = rf"{HF_HOME}/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/tokenizer"
text_encoder_path = rf"{HF_HOME}/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/text_encoder"
# print(torch.__version__)
# print(torch.cuda.is_available())
pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(pipeline_path)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()
pipeline.safety_checker = None
dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_path)
dinov2_model = Dinov2Model.from_pretrained(dinov2_path).to("cuda")
clip_processor = CLIPFeatureExtractor.from_pretrained(clip_path)
clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_path).to("cuda")
text_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
text_model = CLIPTextModel.from_pretrained(text_encoder_path).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(123456789)


# %%
def get_embeddings(processor, model, inputs, output_type="last_hidden_state"):
    if isinstance(processor, CLIPTokenizer):
        features = processor(
            inputs, return_tensors="pt", padding="max_length", max_length=processor.model_max_length
        ).to("cuda")
    else:
        features = processor(inputs, return_tensors="pt").to("cuda")
    outputs = model(**features, return_dict=True)
    embeddings = outputs[output_type]
    if embeddings.ndim < 3:
        embeddings = embeddings.unsqueeze(1)
    return embeddings


def generate(
    pipeline,
    prompts,
    lora_path=None,
    ip_adapter_path=None,
    processor=None,
    image_encoder=None,
    ref_images=None,
    seed=666666,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipeline.unload_ip_adapter()
    pipeline.unload_lora_weights()
    # outputs_1 = pipeline(prompt=prompts, generator=generator).images
    outputs_1 = [Image.open("output_0.png")]
    if lora_path is not None:
        pipeline.load_lora_weights(lora_path)
    if ip_adapter_path is not None:
        ip_adapter_sd = torch.load(ip_adapter_path)
        load_ip_adapter_to_unet(pipeline.unet, ip_adapter_sd)
    if processor is not None and image_encoder is not None and ref_images is not None:
        ip_adapter_embeds = get_embeddings(processor, image_encoder, ref_images)
        image_size = 1024
        zero_img = Image.new("RGB", (image_size, image_size))
        negative_ip_adapter_embeds = get_embeddings(processor, image_encoder, zero_img)
        negative_ip_adapter_embeds = negative_ip_adapter_embeds.repeat(
            ip_adapter_embeds.shape[0], *([1] * len(ip_adapter_embeds.shape[1:]))
        )
        ip_adapter_embeds = torch.cat([negative_ip_adapter_embeds, ip_adapter_embeds])
    else:
        ip_adapter_embeds = None
    prompt_embeds = None
    negative_prompt_embeds = None
    # prompt_embeds = get_embeddings(text_tokenizer, text_model, prompts)
    # negative_prompt_embeds = get_embeddings(text_tokenizer, text_model, [""])
    outputs_2 = pipeline(
        prompt=None if prompt_embeds is not None else prompts,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds=prompt_embeds,
        ip_adapter_image_embeds=[ip_adapter_embeds] if ip_adapter_embeds is not None else None,
        generator=generator,
    ).images
    return outputs_1 + outputs_2


prompts = ["photo,liuyin,person"]
# prompts = ["person"]
lora_path = r"/mnt/d/workspace/Output/diffusers-lora-v2/lora_weights.pth"
ip_adapter_path = r"/mnt/d/workspace/Output/train_dinov2_adapter/checkpoint-4000/dinov2_encoder-4000.pth"
ref_path = r"./dataset/train/images/person/person-1.png"

ref_images = [Image.open(ref_path).convert("RGB")]
outputs = generate(
    pipeline,
    prompts,
    lora_path=None,
    ip_adapter_path=ip_adapter_path,
    processor=dinov2_processor,
    image_encoder=dinov2_model,
    ref_images=ref_images,
    seed=1234,
)
for i, output in enumerate(outputs):
    output.save(f"output_{i}.png")
# %%
