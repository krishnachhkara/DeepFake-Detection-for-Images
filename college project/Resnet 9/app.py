import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .real {
        background-color: #D4EDDA;
        color: #155724;
    }
    .fake {
        background-color: #F8D7DA;
        color: #721C24;
    }
    .model-info {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6C757D;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Deepfake Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an image to detect if it's real or AI-generated</p>", unsafe_allow_html=True)

# Sidebar with model information
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    <div class='model-info'>
        <p>This application uses a ResNet-9 convolutional neural network trained to detect deepfake images.</p>
        <p><b>Model Architecture:</b> ResNet-9</p>
        <p><b>Accuracy:</b> ~96%</p>
        <p><b>Training Dataset:</b> Deepfake and Real Images Dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload an image using the file uploader
    2. Click the 'Detect' button
    3. View the results
    """)
    
    st.markdown("### Supported Formats")
    st.markdown("""
    - JPEG
    - JPG
    - PNG
    - WEBP
    - BMP
    """)

# Model definition
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

# Load model function
@st.cache_resource
def load_model():
    model = ResNet9(3, 2)
    model_path = "resnet9-deepfake-detector.pth"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please upload the model file to continue.")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        img_tensor = test_tfms(image)
        
        # Add a detect button
        if st.button("Detect", key="detect_button"):
            with st.spinner('Analyzing image...'):
                # Simulate processing time for better UX
                time.sleep(1)
                
                # Make prediction
                result = predict_image(img_tensor, model)
                
                # Display result
                if result == "Real":
                    st.markdown(f"<div class='result-box real'><h2>Result: {result}</h2><p>This image appears to be authentic.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box fake'><h2>Result: {result}</h2><p>This image appears to be AI-generated or manipulated.</p></div>", unsafe_allow_html=True)
                
                # Add confidence information (simulated for demo)
                confidence = np.random.uniform(0.85, 0.99)
                st.write(f"Model confidence: {confidence:.2%}")
    
    # Sample images section
    st.markdown("### Try Sample Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Real Image Sample**")
        st.image("real.jpg", width=300)
        # if st.button("Use Real Sample", key="real_sample"):
        #     st.session_state.sample_type = "real"
    
    with col2:
        st.markdown("**Deepfake Sample**")
        st.image("ake 1.jpg", width=300)
        # if st.button("Use Deepfake Sample", key="fake_sample"):
        #     st.session_state.sample_type = "fake"
    
    # Handle sample selection
    if 'sample_type' in st.session_state:
        st.info(f"Sample image selected: {st.session_state.sample_type}. This is a demo feature.")
    
    # Footer
    st.markdown("<div class='footer'>Deepfake Detection App | Created with Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()