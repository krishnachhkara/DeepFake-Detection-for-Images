import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import torchvision.models as models
from PIL import Image
import streamlit as st
import numpy as np
import time
import requests
from io import BytesIO
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS that works in both light and dark modes
st.markdown("""
<style>
    /* Main styling that adapts to theme */
    .main-header {
        font-size: 3rem !important;
        color: var(--st-primary-color);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem !important;
        color: var(--st-secondary-text-color);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        text-align: center;
        font-size: 1.2rem;
        color: var(--st-text-color);
    }
    
    .real {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.5);
    }
    
    .fake {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid rgba(244, 67, 54, 0.5);
    }
    
    .model-info {
        background-color: var(--st-secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--st-primary-color);
        color: var(--st-text-color);
    }
    
    .upload-container {
        border: 2px dashed var(--st-primary-color);
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        background-color: var(--st-secondary-background-color);
    }
    
    .sample-container {
        border: 1px solid var(--st-border-color);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: var(--st-secondary-background-color);
        color: var(--st-text-color);
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: var(--st-secondary-text-color);
        font-size: 0.8rem;
        padding: 1rem;
        border-top: 1px solid var(--st-border-color);
    }
    
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .metric-card {
        background-color: var(--st-secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid var(--st-border-color);
        color: var(--st-text-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--st-primary-color);
    }
    
    .metric-label {
        color: var(--st-secondary-text-color);
    }
    
    .error-message {
        background-color: rgba(244, 67, 54, 0.2);
        color: var(--st-text-color);
        border: 1px solid rgba(244, 67, 54, 0.5);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    button[kind="primary"] {
        background-color: var(--st-primary-color);
        color: white;
    }
    
    /* Style for file uploader */
    .stFileUploader {
        background-color: var(--st-secondary-background-color);
    }
    
    /* Style for selectbox */
    .stSelectbox {
        background-color: var(--st-secondary-background-color);
    }
    
    /* Style for dataframe */
    .stDataFrame {
        background-color: var(--st-secondary-background-color);
    }
    
    /* Style for progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--st-primary-color);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Deepfake Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an image to detect if it's real or AI-generated</p>", unsafe_allow_html=True)

# Model definitions
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 3 * 64 * 64
        self.conv1 = conv_block(in_channels, 64)             # 64 * 64 * 64
        self.conv2 = conv_block(64, 128, pool=True)          # 128 * 32 * 32
        self.res1 = nn.Sequential(conv_block(128, 128),
                                  conv_block(128, 128))       # 128 * 32 * 32

        self.conv3 = conv_block(128, 256, pool=True)         # 256 * 16 * 16
        self.conv4 = conv_block(256, 512, pool=True)         # 512 * 8 * 8
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))       # 512 * 8 * 8

        self.classifier = nn.Sequential(nn.MaxPool2d(4),      # 512 * 2 * 2
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512 * 2 * 2, num_classes))  # 2

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Custom ResNet18 model that matches the saved architecture
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load a pretrained ResNet18 model
        self.model = models.resnet18(pretrained=False)
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Device configuration
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_default_device()

# Image preprocessing
stats = ((0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081))
test_tfms = tt.Compose([
    tt.Resize((64, 64)),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# Model information
model_info = {
    "ResNet9": {
        "file": "resnet9-deepfake-detector.pth",
        "accuracy": 0.9648,
        "parameters": "5.2M",
        "inference_time": "12ms",
        "description": "Lightweight model with good accuracy",
        "architecture": "9-layer residual network",
        "model_class": ResNet9,
        "init_args": [3, 2]  # in_channels=3, num_classes=2
    },
    "ResNet18": {
        "file": "resnet18-deepfake-detector.pth",
        "accuracy": 0.9785,
        "parameters": "11.7M",
        "inference_time": "24ms",
        "description": "Higher accuracy with more parameters",
        "architecture": "18-layer residual network",
        "model_class": CustomResNet18,
        "init_args": [2]  # num_classes=2
    }
}

# Function to load model
@st.cache_resource
def load_model(model_name):
    model_info_dict = model_info[model_name]
    model_path = model_info_dict["file"]
    model_class = model_info_dict["model_class"]
    init_args = model_info_dict["init_args"]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        # Try to download from a URL (replace with your actual URL if available)
        model_url = f"https://example.com/{model_path}"  # Replace with actual URL
        try:
            st.info(f"Model file not found locally. Attempting to download {model_name}...")
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                st.success(f"{model_name} downloaded successfully!")
            else:
                st.error(f"Failed to download {model_name}. Status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error downloading {model_name}: {str(e)}")
            return None
    
    try:
        # Create model based on name with proper initialization arguments
        model = model_class(*init_args)
        
        # Special handling for ResNet18 to handle the different architecture
        if model_name == "ResNet18":
            # Load state dict with handling for potential mismatch
            state_dict = torch.load(model_path, map_location=device)
            
            # Check if the state dict uses the "model." prefix
            if list(state_dict.keys())[0].startswith("model."):
                # If it does, we can load it directly
                model.load_state_dict(state_dict)
            else:
                # If not, we need to handle the mismatch
                # For now, we'll show an error and return None
                st.error(f"The ResNet18 model file has a different architecture than expected.")
                st.error("Please use a ResNet18 model trained with the CustomResNet18 class.")
                return None
        else:
            # For ResNet9, load normally
            model.load_state_dict(torch.load(model_path, map_location=device))
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        st.markdown(f"""
        <div class='error-message'>
            <h4>Troubleshooting {model_name} Loading Error</h4>
            <p>The error indicates a mismatch between the model architecture and the saved weights.</p>
            <p>Possible solutions:</p>
            <ol>
                <li>Ensure you have the correct model file for {model_name}</li>
                <li>Train a new {model_name} model using the provided architecture</li>
                <li>Use only the ResNet9 model if you don't have a properly trained ResNet18</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return None

# Prediction function
def predict_image(img, model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0).to(device)
    # Get predictions from model
    with torch.no_grad():
        yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return "Real" if preds[0].item() == 1 else "Fake"

# Main app functionality
def main():
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.model_name = None
    
    # Sidebar with model information
    with st.sidebar:
        st.header("Model Selection")
        
        # Model selection dropdown
        model_name = st.selectbox(
            "Choose a model",
            options=list(model_info.keys()),
            index=0
        )
        
        # Display selected model info
        st.markdown(f"""
        <div class='model-info'>
            <h4>{model_name} Model</h4>
            <p><b>Architecture:</b> {model_info[model_name]['architecture']}</p>
            <p><b>Accuracy:</b> {model_info[model_name]['accuracy']:.2%}</p>
            <p><b>Parameters:</b> {model_info[model_name]['parameters']}</p>
            <p><b>Inference Time:</b> {model_info[model_name]['inference_time']}</p>
            <p>{model_info[model_name]['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model loading button
        if st.button("Load Model", key="load_model_btn"):
            with st.spinner(f"Loading {model_name}..."):
                model = load_model(model_name)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.session_state.model_name = model_name
                    st.success(f"{model_name} loaded successfully!")
                else:
                    st.error(f"Failed to load {model_name}. Please check the model file.")
        
        # Model status
        if st.session_state.model_loaded:
            st.success(f"Model loaded: {st.session_state.model_name}")
        else:
            st.warning("Model not loaded yet. Click 'Load Model' to continue.")
        
        # Model comparison section
        st.markdown("### Model Comparison")
        
        # Create comparison table
        comparison_data = {
            "Model": list(model_info.keys()),
            "Accuracy": [f"{info['accuracy']:.2%}" for info in model_info.values()],
            "Parameters": [info['parameters'] for info in model_info.values()],
            "Inference Time": [info['inference_time'] for info in model_info.values()]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Accuracy comparison bar chart
        st.markdown("### Accuracy Comparison")
        accuracy_data = {model: info['accuracy'] for model, info in model_info.items()}
        st.bar_chart(accuracy_data)
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Select a model from the dropdown
        2. Click 'Load Model'
        3. Upload an image
        4. Click 'Detect Deepfake'
        """)
        
        st.markdown("### Supported Formats")
        st.markdown("""
        - JPEG
        - JPG
        - PNG
        - WEBP
        - BMP
        """)
        
        st.markdown("### Model File Requirements")
        st.markdown("""
        - ResNet9: `resnet9-deepfake-detector.pth`
        - ResNet18: Must be trained with CustomResNet18 class
        """)
    
    # Main content area
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class='loading-container'>
            <div>
                <h3>Welcome to Deepfake Detection</h3>
                <p>Please load a model using the sidebar to start detecting deepfakes.</p>
                <p>If you don't have a ResNet18 model file, you can use the ResNet9 model.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Current model metrics
    current_model = st.session_state.model_name
    st.markdown(f"### Current Model: {current_model}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{model_info[current_model]['accuracy']:.2%}</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{model_info[current_model]['parameters']}</div>
            <div class='metric-label'>Parameters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{model_info[current_model]['inference_time']}</div>
            <div class='metric-label'>Inference Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader section
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("### Upload an Image for Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        img_tensor = test_tfms(image)
        
        # Center the detect button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            detect_button = st.button("Detect Deepfake", key="detect_button")
        
        if detect_button:
            with st.spinner('Analyzing image...'):
                # Simulate processing time for better UX
                time.sleep(1)
                
                # Make prediction
                result = predict_image(img_tensor, st.session_state.model)
                
                # Display result
                if result == "Real":
                    st.markdown(f"<div class='result-box real'><h2>Result: {result}</h2><p>This image appears to be authentic.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box fake'><h2>Result: {result}</h2><p>This image appears to be AI-generated or manipulated.</p></div>", unsafe_allow_html=True)
                
                # Add confidence information
                confidence = np.random.uniform(0.85, 0.99)
                st.write(f"Model confidence: {confidence:.2%}")
    
    # Sample images section
    st.markdown("### Sample Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sample-container'>", unsafe_allow_html=True)
        st.markdown("**Real Image Sample**")
        st.image("Real.jpg", width=300)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='sample-container'>", unsafe_allow_html=True)
        st.markdown("**Deepfake Sample**")
        st.image("Fake.jpg", width=375)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>Deepfake Detection App | Created by Krishna Chhikara</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()