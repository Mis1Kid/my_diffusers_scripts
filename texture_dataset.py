from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import os
from transformers import CLIPImageProcessor, CLIPTextModel, AutoImageProcessor, Dinov2Model
from PIL import Image
from myutils import prepare_mask_and_masked_image, random_mask
import random
import torch
import itertools


class MyDataset(Dataset):
    def __init__(self, root_dir, tokenizer, feature_extractor, resolution=512, center_crop=True, random_flip=True):
        self.root_dir = Path(root_dir)
        self.transform_resize = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        exts = [".png", ".jpg", ".jpeg"]
        self.train_dir = self.root_dir / "train"
        self.train_images_dir = self.train_dir / "images"/"liuyin"
        self.train_images_path = [self.train_images_dir.rglob(f"*{ext}") for ext in exts]
        self.train_images_path = list(itertools.chain(*self.train_images_path))
        self.image_refs = {}
        for image_path in self.train_images_path:
            image_name = image_path.name
            prefix, postfix = image_name.split("-")
            if prefix not in self.image_refs:
                self.image_refs[prefix] = []
            self.image_refs[prefix].append(image_path)

        self.test_dir = self.root_dir / "test"
        self.test_images_dir = self.test_dir / "images"
        self.test_refs_dir = self.test_dir / "refs"
        self.test_images_path = list(self.test_images_dir.iterdir())
        self.test_refs_path = list(self.test_refs_dir.iterdir())

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.text_prompt_dropout = 0.1
        self.image_prompt_dropout = 0.1
        self.text_image_dropout = 0.1
        self.instance_prompt = "liuyin"
        self.class_prompt = "person"
        self.supplement_prompt = "photo"
        self.all_prompts = [self.instance_prompt, self.class_prompt, self.supplement_prompt]

    def __len__(self):
        return len(self.train_images_path)

    def __getitem__(self, idx):
        image_path = self.train_images_path[idx]
        ref_path = self.get_ref_path(image_path)

        image = Image.open(image_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")

        rate = random.uniform(0, 1)
        text_prompt = ",".join(self.all_prompts)
        image_prompt = ref

        pixel_values = self.transform_resize(image)

        if rate < self.text_prompt_dropout + self.text_image_dropout:
            text_prompt = ""
        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt", padding="max_length", max_length=self.tokenizer.model_max_length
        ).input_ids

        image_features = self.feature_extractor(image_prompt, return_tensors="pt").pixel_values.squeeze(0)

        if (
            rate > self.text_prompt_dropout
            and rate < self.text_prompt_dropout + self.text_image_dropout + self.image_prompt_dropout
        ):
            image_features = torch.zeros_like(image_features)

        example = {}
        example["input_ids"] = input_ids
        example["image_features"] = image_features
        example["pixel_values"] = pixel_values
        return example

    def get_ref_path(self, image_path):
        image_name = image_path.name
        prefix, postfix = image_name.split("-")
        same_prefix_refs = self.image_refs[prefix]
        return random.choice(same_prefix_refs)
