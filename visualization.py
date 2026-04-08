import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

def overlay_heatmap(img_pil, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays a 2D float heatmap (0 to 1) on a PIL Image.
    """
    if type(img_pil) is not np.ndarray:
        img_np = np.array(img_pil)
    else:
        img_np = img_pil
        
    if len(img_np.shape) == 2:
         img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Normalize the heatmap between 0 and 1 if it isn't already
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Normalize between 0 and 255 for colormap
    heatmap_norm = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend the original image and the heatmap
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)

def plot_xai_comparison(original_img, resnet_heatmap, vit_heatmap, title="XAI Comparison"):
    """
    Plots the original image alongside the models' explanations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Artwork")
    axes[0].axis('off')
    
    resnet_overlay = overlay_heatmap(original_img, resnet_heatmap)
    axes[1].imshow(resnet_overlay)
    axes[1].set_title("ResNet-18 (Grad-CAM)")
    axes[1].axis('off')
    
    vit_overlay = overlay_heatmap(original_img, vit_heatmap)
    axes[2].imshow(vit_overlay)
    axes[2].set_title("ViT (Attention Rollout)")
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
