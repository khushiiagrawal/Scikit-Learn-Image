import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import joblib
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_all_features
from src.model import BrainTumorClassifier

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI Images")
st.write("Upload an MRI image to check for brain tumor.")

# Add information about image processing techniques
with st.expander("📊 Image Processing Techniques Used"):
    st.markdown("""
    ### Image Preprocessing Pipeline
    This project uses scikit-image for advanced image processing:
    
    1. **Grayscale Conversion**
       - Converts RGB/RGBA images to grayscale using `skimage.color.rgb2gray`
       - Reduces complexity while preserving important features
    
    2. **Image Normalization**
       - Normalizes pixel values to range [0, 1]
       - Ensures consistent intensity levels across different images
    
    3. **Noise Removal**
       - Applies Gaussian blur using `skimage.filters.gaussian`
       - Reduces noise while preserving important edges
    
    4. **Contrast Enhancement**
       - Uses histogram equalization (`skimage.exposure.equalize_hist`)
       - Improves visibility of tumor regions
    
    5. **Image Resizing**
       - Resizes images to 256x256 using `skimage.transform.resize`
       - Ensures consistent input size for feature extraction
    
    ### Feature Extraction
    The following features are extracted using scikit-image:
    
    1. **HOG Features**
       - Histogram of Oriented Gradients
       - Captures shape and edge information
    
    2. **LBP Features**
       - Local Binary Patterns
       - Captures texture information
    
    3. **Texture Features**
       - Uses Sobel operator for edge detection
       - Extracts mean, standard deviation, and maximum edge values
    
    4. **Intensity Features**
       - Mean, standard deviation, and median intensity values
       - Captures overall image characteristics
    """)

# Load model
MODEL_PATH = "brain_tumor_classifier.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found! Please train the model first.")
    st.stop()
model = BrainTumorClassifier()
model.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose an MRI image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert to numpy array and save temporarily
    img_np = np.array(image)
    temp_path = "temp_uploaded_image.png"
    Image.fromarray(img_np).save(temp_path)

    # Preprocess
    preprocessed = preprocess_image(temp_path)
    st.image(preprocessed, caption="Preprocessed Image", use_container_width=True, channels="GRAY")

    # Feature extraction
    features = extract_all_features(preprocessed).reshape(1, -1)

    # Prediction
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"**Tumor Detected!** (Confidence: {proba[1]*100:.1f}%)")
    else:
        st.success(f"**No Tumor Detected.** (Confidence: {proba[0]*100:.1f}%)")

    # Clean up temp file
    os.remove(temp_path)
else:
    st.info("Please upload an MRI image.") 