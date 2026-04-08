import torch
import matplotlib
# Set matplotlib backend to agg to prevent it from trying to open a window on the server
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from xai_utils import load_wikiart_dataset, load_models_and_processors, preprocess_image
from resnet_xai import generate_grad_cam
from vit_xai import generate_vit_rollout
from visualization import plot_xai_comparison

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Load models
        print("Initializing models. This may take a moment to download weights if not cached...")
        (vit_model, vit_processor), (resnet_model, resnet_processor) = load_models_and_processors(device)
        
        # 2. Get data
        print("Fetching a sample image from the Wikiart dataset...")
        images = load_wikiart_dataset(num_samples=1)
        if not images:
            print("Failed to pull any images.")
            return
            
        image = images[0]
        
        # 3. ResNet XAI
        print("Running Grad-CAM on ResNet-18...")
        resnet_inputs = preprocess_image(image, resnet_processor, device)
        resnet_heatmap = generate_grad_cam(resnet_model, resnet_inputs['pixel_values'], target_class=None)
        
        # 4. ViT XAI
        print("Running Attention Rollout on Vision Transformer...")
        vit_inputs = preprocess_image(image, vit_processor, device)
        vit_heatmap = generate_vit_rollout(vit_model, vit_inputs['pixel_values'])
        
        # 5. Visualization and Save
        print("Generating visualization...")
        plot_xai_comparison(image, resnet_heatmap, vit_heatmap, title=f"XAI Analysis on Wikiart Image")
        output_file = "output_xai_result.png"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Success! Saved visualization to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
