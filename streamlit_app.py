import streamlit as st
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from xai_utils import load_models_and_processors, preprocess_image, get_random_wikiart_sample
from resnet_xai import generate_grad_cam, generate_integrated_gradients, generate_occlusion_sensitivity
from vit_xai import generate_vit_rollout
from visualization import overlay_heatmap

st.set_page_config(page_title="Deep Learning Art XAI", layout="wide")
st.title("🎨 Deep Learning Art Interpretability (XAI)")
st.markdown("Upload an artwork or sample one from the WikiArt dataset to see a live XAI comparison between a ResNet CNN and a Vision Transformer.")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_models_and_processors(device), device

(vit_resources, resnet_resources), device = load_models()
vit_model, vit_processor = vit_resources
resnet_model, resnet_processor = resnet_resources

st.sidebar.header("Input Method")
input_option = st.sidebar.radio("Choose how to get an image:", ["Sample from WikiArt", "Upload your own object"])

img_pil = None
if input_option == "Sample from WikiArt":
    if st.sidebar.button("Fetch Random Sample"):
        with st.spinner("Fetching a random sample from huggingface..."):
            sample_img, true_style = get_random_wikiart_sample()
            st.session_state["current_img"] = sample_img
            st.session_state["true_style"] = true_style
            
    if "current_img" in st.session_state:
        img_pil = st.session_state["current_img"]
        st.write(f"**Dataset Label ID:** {st.session_state['true_style']}")

elif input_option == "Upload your own object":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")

if img_pil is not None:
    st.subheader("Original Image")
    st.image(img_pil, use_container_width=False, width=300)

    if st.button("Run XAI Pipelines"):
        with st.spinner("Running models and XAI..."):
            # ResNet Pipeline
            resnet_inputs = preprocess_image(img_pil, resnet_processor, device)
            # Classification
            resnet_outputs = resnet_model(**resnet_inputs)
            resnet_pred_id = resnet_outputs.logits.argmax(-1).item()
            resnet_class_name = resnet_model.config.id2label[resnet_pred_id]
            
            # XAI
            resnet_gradcam = generate_grad_cam(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
            resnet_gradcam_overlay = overlay_heatmap(img_pil, resnet_gradcam)
            
            resnet_ig = generate_integrated_gradients(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
            resnet_ig_overlay = overlay_heatmap(img_pil, resnet_ig)
            
            resnet_occlusion = generate_occlusion_sensitivity(resnet_model, resnet_inputs['pixel_values'], target_class=resnet_pred_id)
            resnet_occlusion_overlay = overlay_heatmap(img_pil, resnet_occlusion)
            
            # ViT Pipeline
            vit_inputs = preprocess_image(img_pil, vit_processor, device)
            # Classification
            vit_outputs = vit_model(**vit_inputs)
            vit_pred_id = vit_outputs.logits.argmax(-1).item()
            vit_class_name = vit_model.config.id2label[vit_pred_id]
            
            # XAI
            vit_rollout = generate_vit_rollout(vit_model, vit_inputs['pixel_values'])
            vit_rollout_overlay = overlay_heatmap(img_pil, vit_rollout)
            
            # Display
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("ResNet-18 Analysis")
                st.success(f"**Predicted Style:** {resnet_class_name} (ID: {resnet_pred_id})")
                
                st.subheader("Grad-CAM")
                st.image(resnet_gradcam_overlay, caption="Where the model looked (Grad-CAM)")
                
                st.subheader("Integrated Gradients")
                st.image(resnet_ig_overlay, caption="Pixel attribution map (Integrated Gradients)")
                
                st.subheader("Occlusion Sensitivity")
                st.image(resnet_occlusion_overlay, caption="Occlusion sensitivity map")
                
            with col2:
                st.header("Vision Transformer Analysis")
                st.success(f"**Predicted Style:** {vit_class_name} (ID: {vit_pred_id})")
                
                st.subheader("Attention Rollout")
                st.image(vit_rollout_overlay, caption="Information flow & attention via Rollout")
