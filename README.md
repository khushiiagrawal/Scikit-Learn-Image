# Brain Tumor Detection using Image Processing

This project uses scikit-image and machine learning techniques to detect brain tumors in MRI images.

## Project Structure
- `data/` - Directory for storing MRI images
- `src/` - Source code directory
  - `preprocessing.py` - Image preprocessing functions
  - `feature_extraction.py` - Feature extraction functions
  - `model.py` - Machine learning model implementation
  - `main.py` - Main script to run the project

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
mkdir data src
```

## Data Requirements
- MRI images in DICOM or PNG format
- Images should be organized in folders:
  - `data/train/` - Training images
  - `data/test/` - Test images
  - `data/validation/` - Validation images

## Usage
1. Place your MRI images in the appropriate directories
2. Run the main script:
```bash
python src/main.py
```

## Features
- Image preprocessing (noise removal, normalization)
- Feature extraction
- Tumor detection using machine learning
- Visualization of results 