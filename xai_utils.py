import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from PIL import Image

# Art style to era mapping
STYLE_TO_ERA = {
    # Renaissance (1400-1600)
    "Early Renaissance": "Renaissance",
    "High Renaissance": "Renaissance", 
    "Mannerism (Late Renaissance)": "Renaissance",
    
    # Baroque (1600-1750)
    "Baroque": "Baroque",
    "Dutch Golden Age": "Baroque",
    
    # Rococo (1700-1780)
    "Rococo": "Rococo",
    
    # Neoclassicism (1750-1850)
    "Neoclassicism": "Neoclassicism",
    
    # Romanticism (1780-1850)
    "Romanticism": "Romanticism",
    
    # Realism (1840-1880)
    "Realism": "Realism",
    
    # Impressionism (1860-1890)
    "Impressionism": "Impressionism",
    "Post-Impressionism": "Post-Impressionism",
    
    # Modern Art (1880-1970)
    "Art Nouveau (Modern)": "Modern Art",
    "Expressionism": "Modern Art",
    "Cubism": "Modern Art",
    "Futurism": "Modern Art",
    "Dadaism": "Modern Art",
    "Surrealism": "Modern Art",
    "Abstract Expressionism": "Modern Art",
    "Pop Art": "Modern Art",
    "Minimalism": "Modern Art",
    
    # Contemporary (1970-present)
    "Contemporary Realism": "Contemporary",
    "Street Art": "Contemporary",
    "Digital Art": "Contemporary",
    
    # Other mappings
    "Northern Renaissance": "Renaissance",
    "Mannerism": "Renaissance",
    "Academicism": "Academic Art",
    "Symbolism": "Symbolism",
    "Pointillism": "Post-Impressionism",
    "Fauvism": "Modern Art",
    "Constructivism": "Modern Art",
    "Suprematism": "Modern Art",
    "Bauhaus": "Modern Art",
    "Art Deco": "Modern Art",
    "Op Art": "Modern Art",
    "Conceptual Art": "Contemporary",
    "Installation Art": "Contemporary",
    "Performance Art": "Contemporary",
}

def get_era_from_style(style_name):
    """
    Map art style to historical era.
    """
    return STYLE_TO_ERA.get(style_name, "Unknown Era")

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
