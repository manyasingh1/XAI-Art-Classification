import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from captum.attr import IntegratedGradients

class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingFaceModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, pixel_values):
        # The Grad-CAM library expects the forward pass to return the raw logits directly
        # Hugging Face models return a specific class like ImageClassificationOutput
        return self.model(pixel_values).logits

def get_resnet_target_layer(model):
    """
    Automatically tries to locate the final convolutional layer in the Hugging Face ResNet architecture.
    """
    try:
        # For standard transformers ResNetForImageClassification
        return [model.resnet.encoder.stages[-1].layers[-1]]
    except AttributeError:
        # Fallback if structure varies
        print("Warning: Could not automatically detect standard format target layer. Falling back to simple heuristic.")
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
        return None

def generate_grad_cam(model, image_tensor, target_class=None):
    """
    Generates a Grad-CAM heatmap for a given image tensor using the provided ResNet model.
    """
    target_layers = get_resnet_target_layer(model)
    if target_layers is None:
        raise ValueError("Could not find valid target layer for GradCAM.")
    
    wrapper = HuggingFaceModelWrapper(model)
    
    # We construct the GradCAM object
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    
    targets = None
    if target_class is not None:
         targets = [ClassifierOutputTarget(target_class)]
         
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam[0, :]

def generate_integrated_gradients(model, image_tensor, target_class):
    """
    Generates a pixel-attribution map using Integrated Gradients.
    """
    wrapper = HuggingFaceModelWrapper(model)
    ig = IntegratedGradients(wrapper)
    
    # Ensure inputs require grad
    image_tensor.requires_grad_()
    
    attr, delta = ig.attribute(image_tensor, target=target_class, return_convergence_delta=True)
    
    attr = attr.squeeze().cpu().detach().numpy()
    
    # Average attribution across color channels
    attr = np.mean(attr, axis=0)
    # Consider absolute magnitude
    attr = np.abs(attr)
    
    # Normalize between 0 and 1
    if np.max(attr) > 0:
        attr = attr / np.max(attr)
        
    return attr

def generate_occlusion_sensitivity(model, image_tensor, target_class, patch_size=16, stride=8):
    """
    Generates an occlusion sensitivity map by masking patches of the image and measuring prediction change.
    """
    model.eval()
    with torch.no_grad():
        original_output = model(image_tensor)
        original_prob = torch.softmax(original_output.logits, dim=1)[0, target_class].item()
    
    _, _, H, W = image_tensor.shape
    heatmap = np.zeros((H, W))
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            # Create occluded image
            occluded = image_tensor.clone()
            occluded[:, :, i:i+patch_size, j:j+patch_size] = 0  # Black out patch
            
            with torch.no_grad():
                occluded_output = model(occluded)
                occluded_prob = torch.softmax(occluded_output.logits, dim=1)[0, target_class].item()
            
            # Sensitivity: how much the probability drops
            sensitivity = original_prob - occluded_prob
            heatmap[i:i+patch_size, j:j+patch_size] += sensitivity
    
    # Normalize
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap
