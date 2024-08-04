import torch
from PIL import Image
import numpy as np
from transformers import (
    BitImageProcessor,
    Dinov2Model,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from pathlib import Path
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
import seaborn as sns

clip_dir = r"openai/clip-vit-large-patch14"
dinov2_dir = r"facebook/dinov2-base"
blip2_dir = r"Salesforce/blip-image-captioning-large"
image_dir = Path(r"/mnt/d/workspace/Datasets/TextureInpaint/train/images")
image_pathes = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
images = [Image.open(image_path).convert("RGB") for image_path in image_pathes]

dinov2_processor = BitImageProcessor.from_pretrained(dinov2_dir)
dinov2_model = Dinov2Model.from_pretrained(dinov2_dir).to("cuda")

clip_processor = CLIPImageProcessor.from_pretrained(clip_dir)
clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_dir).to("cuda")

processor = BlipProcessor.from_pretrained(blip2_dir)
model = BlipForConditionalGeneration.from_pretrained(blip2_dir).to("cuda")


def get_embedding(processor, model, images):
    features = processor(images, return_tensors="pt").to("cuda")
    outputs = model(**features)
    if "pooler_output" in outputs.keys():
        embeddings = outputs.pooler_output
    if "image_embeds" in outputs.keys():
        embeddings = outputs.image_embeds
    return embeddings.detach().cpu().numpy()


dinov2_embeddings = get_embedding(dinov2_processor, dinov2_model, images)
clip_embeddings = get_embedding(clip_processor, clip_model, images)

dinov2_scores = [[0] * len(images) for _ in range(len(images))]
clip_scores = [[0] * len(images) for _ in range(len(images))]
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        dinov2_similarity = 1 - cosine(dinov2_embeddings[i], dinov2_embeddings[j])
        clip_similarity = 1 - cosine(clip_embeddings[i], clip_embeddings[j])
        dinov2_scores[i][j] = dinov2_scores[j][i] = dinov2_similarity
        clip_scores[i][j] = clip_scores[j][i] = clip_similarity


plt.subplot(1, 2, 1)
plt.title("DINOv2")
sns.heatmap(dinov2_scores, annot=True, fmt=".2f", cmap="binary")
plt.subplot(1, 2, 2)
plt.title("CLIP")
sns.heatmap(clip_scores, annot=True, fmt=".2f", cmap="binary")
plt.show()
