import torch
import cv2
import numpy as np

def generate_vit_attention(model, image_tensor, target_layer=-1):
    """
    Extracts raw attention from a specific layer of the ViT.
    Returns the attention of the CLS token towards all other patches for the given layer.
    """
    outputs = model(image_tensor, output_attentions=True)
    attentions = outputs.attentions  # Tuple of (batch, heads, seq_len, seq_len)
    
    # Take the specified layer's attention (e.g. final layer)
    attn = attentions[target_layer][0] # shape: (heads, seq_len, seq_len)
    
    # Average across heads
    attn_mean = attn.mean(dim=0) # shape: (seq_len, seq_len)
    
    # Take attention from the CLS token (index 0) to all other image patches (index 1:)
    cls_attention = attn_mean[0, 1:]
    
    # Assuming standard ViT with 14x14 patches (196 layout + 1 cls)
    grid_size = int(np.sqrt(cls_attention.size(0)))
    heatmap = cls_attention.reshape(grid_size, grid_size).detach().cpu().numpy()
    
    return heatmap

def attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Computes Attention Rollout to get the relevancy across all layers.
    """
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    with torch.no_grad():
        for attention in attentions:
            attention = attention.detach()
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type Not supported")

            # Add identity matrix to account for skip connections
            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
            
    # Take the CLS token's view of the rest of the image patches
    # result shape for single batch: [1, seq_len, seq_len], we take [0, 0, 1:]
    mask = result[0, 0, 1:]
    
    # Normalize between 0 and 1
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    grid_size = int(np.sqrt(mask.size(0)))
    mask = mask.reshape(grid_size, grid_size).cpu().numpy()
    return mask

def generate_vit_rollout(model, image_tensor):
    """
    Wrapper for computing Attention Rollout for a full model run.
    """
    outputs = model(image_tensor, output_attentions=True)
    attentions = outputs.attentions
    
    heatmap = attention_rollout(attentions)
    return heatmap
