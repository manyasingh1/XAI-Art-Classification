import torch
from xai_utils import load_models_and_processors, preprocess_image, get_random_wikiart_sample
from resnet_xai import generate_grad_cam, generate_integrated_gradients, generate_occlusion_sensitivity
from vit_xai import generate_vit_rollout
from visualization import overlay_heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
(vit_model, vit_processor), (resnet_model, resnet_processor) = load_models_and_processors(device)

# Load a sample image
image, true_style = get_random_wikiart_sample()
print(f"Loaded image with style: {true_style}")

# Preprocess for ResNet
resnet_inputs = preprocess_image(image, resnet_processor, device)
resnet_outputs = resnet_model(**resnet_inputs)
resnet_pred_id = resnet_outputs.logits.argmax(-1).item()
resnet_class_name = resnet_model.config.id2label[resnet_pred_id]
print(f"ResNet prediction: {resnet_class_name}")

# Grad-CAM
resnet_gradcam = generate_grad_cam(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
print(f"Grad-CAM shape: {resnet_gradcam.shape}")

# Integrated Gradients
resnet_ig = generate_integrated_gradients(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
print(f"IG shape: {resnet_ig.shape}")

# Occlusion
resnet_occlusion = generate_occlusion_sensitivity(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
print(f"Occlusion shape: {resnet_occlusion.shape}")

# ViT
vit_inputs = preprocess_image(image, vit_processor, device)
vit_outputs = vit_model(**vit_inputs)
vit_pred_id = vit_outputs.logits.argmax(-1).item()
vit_class_name = vit_model.config.id2label[vit_pred_id]
print(f"ViT prediction: {vit_class_name}")

# Attention Rollout
vit_rollout = generate_vit_rollout(vit_model, vit_inputs['pixel_values'])
print(f"Rollout shape: {vit_rollout.shape}")

print("All XAI methods executed successfully!")