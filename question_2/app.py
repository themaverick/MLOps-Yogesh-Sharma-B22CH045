import streamlit as st
import torch
import torch.nn as nn
from utils import get_model, UNet
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Segmentation Model Dashboard", layout="wide")

CHECKPOINT_PATH = "checkpoint.pth"
MODEL_PATH = "model_final.pth"
DATA_DIR = "data"
RGB_DIR = os.path.join(DATA_DIR, "CameraRGB")
MASK_DIR = os.path.join(DATA_DIR, "CameraMask")

@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint for metrics
    checkpoint = None
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Load model
    model = get_model(n_classes=23)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    elif checkpoint and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    model.eval()
    
    return model, checkpoint, device

model, checkpoint, device = load_resources()

def get_color_map(n_classes=23):
    cmap = plt.get_cmap("tab20", n_classes)
    colors = [cmap(i)[:3] for i in range(n_classes)]
    return (np.array(colors) * 255).astype(np.uint8)

COLOR_MAP = get_color_map(23)

def decode_mask(mask, colormap):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(len(colormap)):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training Metrics", "Model Inference"])

# --- Page 1: Training Metrics ---
if page == "Training Metrics":
    st.title("📊 Training Phase Metrics")
    
    if checkpoint:
        train_losses = checkpoint.get("train_losses", [])
        val_ious = checkpoint.get("val_ious", [])
        val_dices = checkpoint.get("val_dices", [])
        epochs = range(1, len(train_losses) + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Curve")
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(epochs, train_losses, label="Train Loss", marker='o')
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)
            
        with col2:
            st.subheader("Validation Metrics")
            fig_met, ax_met = plt.subplots()
            ax_met.plot(epochs, val_ious, label="mIOU", marker='s')
            ax_met.plot(epochs, val_dices, label="mDice", marker='^')
            ax_met.set_xlabel("Epoch")
            ax_met.set_ylabel("Score")
            ax_met.legend()
            st.pyplot(fig_met)
            
        st.divider()
        st.subheader("Final Test Set Results")
        if val_ious and val_dices:
            st.metric("Test mIOU", f"{val_ious[-1]:.4f}")
            st.metric("Test mDice", f"{val_dices[-1]:.4f}")
    else:
        st.warning("Checkpoint file not found. Training metrics cannot be displayed.")
        # Fallback values if hardcoded metrics are needed
        st.info("Showing example metrics from notebook output:")
        st.write("- **Epoch 15**: Loss=0.0717, mIoU=0.8086, mDice=0.8693")

elif page == "Model Inference":
    st.title("🔍 Model Prediction")
    st.write("Upload up to 4 images from the test set for segmentation.")

    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 4:
            st.warning("Please upload a maximum of 4 images.")
            uploaded_files = uploaded_files[:4]
            
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.subheader(f"Image: {file_name}")
            
            # Read Image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run Inference
            h, w = image_rgb.shape[:2]
            input_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            pred_mask_rgb = decode_mask(pred_mask, COLOR_MAP)
            
            gt_mask_rgb = None
            if os.path.exists(os.path.join(MASK_DIR, file_name)):
                gt_mask = cv2.imread(os.path.join(MASK_DIR, file_name), 0)
                gt_mask_rgb = decode_mask(gt_mask, COLOR_MAP)
            
            # Display Results
            cols = st.columns(3 if gt_mask_rgb is not None else 2)
            with cols[0]:
                st.image(image_rgb, caption="Original Image")
            if gt_mask_rgb is not None:
                with cols[1]:
                    st.image(gt_mask_rgb, caption="Ground Truth Mask")
                with cols[2]:
                    st.image(pred_mask_rgb, caption="Predicted Mask")
            else:
                with cols[1]:
                    st.image(pred_mask_rgb, caption="Predicted Mask")
            
            st.divider()
