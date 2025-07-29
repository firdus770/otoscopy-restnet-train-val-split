import streamlit as st
import os
import subprocess
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

# --- Google Drive Model Setup ---
MODEL_PATH = "resnet18_otoscopic.pt"
MODEL_ID = "1s4GKO6j5JfbrSsafNXj0lodyNJCQOmmZ"  # Your actual Google Drive file ID

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    try:
        import gdown
    except ImportError:
        subprocess.run(["pip", "install", "gdown"])
        import gdown

    subprocess.run(["gdown", "--id", MODEL_ID, "--output", MODEL_PATH], check=True)
    st.success("Model downloaded successfully.")

# --- Device & Labels ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

# --- Load ResNet18 Model ---
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(class_labels))
resnet_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
resnet_model.eval().to(device)

# --- Grad-CAM Setup ---
cam_extractor = GradCAM(resnet_model, target_layer="layer4")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Streamlit UI ---
st.title("Otoscopic Image Classifier with Grad-CAM (ResNet18)")

uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Inference
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = resnet_model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    st.success(f"Prediction: {class_labels[pred_class]}")
    st.info(f"Confidence: {confidence:.2f}")

    # Grad-CAM
    activation_map = cam_extractor(pred_class, output)[0].detach().cpu()
    heatmap = to_pil_image(activation_map)
    heatmap_resized = heatmap.resize(image.size)
    cam_result = overlay_mask(image, heatmap_resized, alpha=0.5)

    st.image(cam_result, caption="Grad-CAM Heatmap", use_container_width=True)
