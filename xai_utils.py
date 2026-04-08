import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from PIL import Image

def load_wikiart_dataset(num_samples=200):
    """
    Loads a streaming portion of the huggan/wikiart dataset.
    """
    print("Loading Wikiart dataset...")
    dataset = load_dataset("huggan/wikiart", streaming=True)
    test_dataset = dataset["train"].take(num_samples)
    
    # Materialize explicitly into a list of PIL images
    images = [example["image"].convert("RGB") for example in test_dataset if example.get("image")]
    print(f"Loaded {len(images)} valid images.")
    return images

def get_random_wikiart_sample():
    """
    Returns a single random image and its style label from the wikiart dataset.
    """
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    import random
    skip_amount = random.randint(0, 1000)
    
    # Skip a random amount to get a random-ish sample from streaming dataset
    iterator = iter(dataset)
    for _ in range(skip_amount):
        next(iterator)
        
    sample = next(iterator)
    # The dataset provides 'style' as an integer. We will return the raw sample.
    return sample["image"].convert("RGB"), sample["style"]


def load_models_and_processors(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads both models and their processors from Hugging Face.
    """
    print(f"Loading models to {device}...")

    vit_name = "sakhmatd/vit-wikiart-finetuned"
    resnet_name = "Iust1n2/resnet-18-finetuned-wikiart"

    # We need to set output_attentions=True to extract attention weights for ViT
    vit_model = AutoModelForImageClassification.from_pretrained(vit_name, output_attentions=True).to(device)
    vit_processor = AutoImageProcessor.from_pretrained(vit_name)

    resnet_model = AutoModelForImageClassification.from_pretrained(resnet_name).to(device)
    resnet_processor = AutoImageProcessor.from_pretrained(resnet_name)

    return (vit_model, vit_processor), (resnet_model, resnet_processor)

def preprocess_image(image, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Preprocesses a PIL Image using the given feature extractor/processor.
    """
    inputs = processor(images=image, return_tensors="pt")
    # Move all input tensors to the target device
    return {k: v.to(device) for k, v in inputs.items()}
