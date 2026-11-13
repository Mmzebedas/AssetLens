
# ---------------------------------
# ASSETLENS - AI ENGINE
# ---------------------------------

import numpy as np
import open_clip
import torch
from PIL import Image

# 1. Model configuration
MODEL_NAME = "ViT-B-32"
WEIGHTS = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512


# 2. Load pretrained CLIP model
def load_clip_model():

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=WEIGHTS,
        device="cpu"
    )

    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    return model, preprocess, tokenizer


# 3. Convert an image into an embedding
def encode_image(model, preprocess, image_path):

    with Image.open(image_path) as img:
        image = preprocess(
            img.convert("RGB")
        ).unsqueeze(0)

    with torch.inference_mode():
        embedding = model.encode_image(
            image,
            normalize=True
        )

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


# 4. Convert text into an embedding
def encode_text(model, tokenizer, query):

    tokens = tokenizer([query])

    with torch.inference_mode():
        embedding = model.encode_text(
            tokens,
            normalize=True
        )

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)
