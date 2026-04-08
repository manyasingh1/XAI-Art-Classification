import gradio as gr
import torch
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

from xai_utils import load_models_and_processors, preprocess_image
from resnet_xai import generate_grad_cam
from vit_xai import generate_vit_rollout
from visualization import overlay_heatmap

print("Booting UP Web Environment...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target compute engine: {device}")

# Mount models exactly once.
(vit_model, vit_processor), (resnet_model, resnet_processor) = load_models_and_processors(device)

def analyze(img_pil):
    if img_pil is None:
        return None, None

    img_pil = img_pil.convert("RGB")
    
    # Run CNN
    resnet_inputs = preprocess_image(img_pil, resnet_processor, device)
    resnet_heatmap = generate_grad_cam(resnet_model, resnet_inputs['pixel_values'], target_class=None)
    resnet_out = overlay_heatmap(img_pil, resnet_heatmap)
    
    # Run Transformer
    vit_inputs = preprocess_image(img_pil, vit_processor, device)
    vit_heatmap = generate_vit_rollout(vit_model, vit_inputs['pixel_values'])
    vit_out = overlay_heatmap(img_pil, vit_heatmap)
    
    return resnet_out, vit_out

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # 🎨 Deep Learning Art Interpretability (XAI)
        Upload an artwork and our custom algorithms will perform a live XAI comparison of what two contrasting state-of-the-art networks look at: a structured CNN vs a spatial Transformer.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Original Input Artwork")
            submit_btn = gr.Button("Run XAI Pipelines", variant="primary")
            
        with gr.Column():
            out_cnn = gr.Image(type="pil", label="ResNet-18 (Grad-CAM)")
            out_vit = gr.Image(type="pil", label="Vision Transformer (Attention Rollout)")
            
    submit_btn.click(analyze, inputs=input_image, outputs=[out_cnn, out_vit])

if __name__ == "__main__":
    print("Hosting demo!")
    demo.launch(inbrowser=True)
