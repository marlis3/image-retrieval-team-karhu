# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" jupyter={"outputs_hidden": true}
# Installing necessary dependencies
# !pip install torch torchvision transformers datasets huggingface_hub kaggle

# +
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import os
from torchvision import transforms

# Load pre-trained CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# +
from pathlib import Path
import numpy as np
dataset_path = Path("/kaggle/input/chexpert/train")

image_paths = np.array(list(dataset_path.rglob("*.jpg")))

print(f" Found {len(image_paths)} images")

# +
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return clip_preprocess(image)

test_image_path = str(image_paths[0])
processed_image = preprocess_image(test_image_path)

print("image shape", processed_image.shape)
