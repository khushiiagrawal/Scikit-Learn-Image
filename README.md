# Brain Tumor Detection using Image Processing

This project uses scikit-image and machine learning techniques to detect brain tumors in MRI images. It includes both a backend model and a user-friendly web interface built with Streamlit.

## Project Structure
```
├── data/                  # Directory for storing MRI images
│   ├── train/            # Training images
│   │   ├── tumor/       # Tumor images for training
│   │   └── no_tumor/    # Non-tumor images for training
│   ├── test/            # Test images
│   └── validation/      # Validation images
├── src/                  # Source code directory
│   ├── preprocessing.py  # Image preprocessing functions
│   ├── feature_extraction.py  # Feature extraction functions
│   ├── model.py         # Machine learning model implementation
│   └── main.py          # Main script to train the model
├── frontend.py          # Streamlit web interface
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Features
- **Image Preprocessing**
  - Grayscale conversion
  - Image normalization
  - Noise removal using Gaussian blur
  - Contrast enhancement using histogram equalization
  - Image resizing

- **Feature Extraction**
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Patterns)
  - Texture features using Sobel operator
  - Intensity-based features

- **Machine Learning**
  - Random Forest Classifier
  - Model training and evaluation
  - Model persistence

- **Web Interface**
  - User-friendly Streamlit frontend
  - Real-time image upload and processing
  - Clear visualization of results
  - Detailed explanation of processing steps

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/train/tumor data/train/no_tumor data/test/tumor data/test/no_tumor data/validation/tumor data/validation/no_tumor src
```

## Data Requirements
- MRI images in DICOM or PNG format
- Images should be organized in folders:
  - `data/train/` - Training images
  - `data/test/` - Test images
  - `data/validation/` - Validation images

## Usage

### Training the Model
1. Place your MRI images in the appropriate directories
2. Run the training script:
```bash
python src/main.py
```

### Using the Web Interface
1. Make sure you have a trained model (`brain_tumor_classifier.joblib`)
2. Run the Streamlit app:
```bash
streamlit run frontend.py
```
3. Open your web browser and navigate to the provided URL (usually http://localhost:8501)
4. Upload an MRI image to get the prediction

## Dependencies
- scikit-image==0.21.0
- numpy==1.24.3
- matplotlib==3.7.1
- scikit-learn==1.3.0
- opencv-python==4.8.0.76
- pandas==2.0.3
- streamlit==1.35.0

## Contributing
Feel free to submit issues and enhancement requests! 