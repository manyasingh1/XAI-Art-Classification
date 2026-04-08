import nbformat as nbf

nb = nbf.v4.new_notebook()

text_md = """# XAI on Wikiart: ResNet vs ViT
This notebook demonstrates Explainable AI techniques by exploring what a Convolutional Neural Network (ResNet-18) and a Vision Transformer (ViT) are paying attention to when classifying artworks.

Both models have been fine-tuned on the Wikiart dataset.

We will use:
- **Grad-CAM** for ResNet-18
- **Attention Rollout** for ViT
"""

code_setup = """%load_ext autoreload
%autoreload 2

import torch
from xai_utils import load_wikiart_dataset, load_models_and_processors, preprocess_image
from resnet_xai import generate_grad_cam
from vit_xai import generate_vit_rollout, generate_vit_attention
from visualization import plot_xai_comparison
"""

text_load = """## 1. Data and Model Loading
We stream images from the `huggan/wikiart` test set, and load our models."""

code_load = """device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
(vit_model, vit_processor), (resnet_model, resnet_processor) = load_models_and_processors(device)

# Load dataset
images = load_wikiart_dataset(num_samples=20)
print(f"Loaded {len(images)} images for testing.")
"""

text_analysis = """## 2. Explainable AI Analysis
Let's analyze the first few images and compare what each model focuses on."""

code_analysis = """# Let's take an example image
img_idx = 0
image = images[img_idx]

# 1. Preprocess for ResNet
resnet_inputs = preprocess_image(image, resnet_processor, device)
# Resnet Grad-CAM
resnet_heatmap = generate_grad_cam(resnet_model, resnet_inputs['pixel_values'], target_class=None)

# 2. Preprocess for ViT
vit_inputs = preprocess_image(image, vit_processor, device)
# ViT Attention Rollout
vit_heatmap = generate_vit_rollout(vit_model, vit_inputs['pixel_values'])

# 3. Visualize Comparison
plot_xai_comparison(image, resnet_heatmap, vit_heatmap, title=f"Analysis on Image {img_idx}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_md),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_analysis),
    nbf.v4.new_code_cell(code_analysis)
]

nbf.write(nb, 'main_analysis.ipynb')
print("Notebook created successfully!")
