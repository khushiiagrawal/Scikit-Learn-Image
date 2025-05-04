import numpy as np
from skimage import io, color, exposure, filters
from skimage.transform import resize
import cv2

def load_image(image_path):
    """Load an image from the given path."""
    image = io.imread(image_path)
    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def convert_to_grayscale(image):
    """Convert image to grayscale if it's RGB."""
    if len(image.shape) == 3:
        return color.rgb2gray(image)
    return image

def normalize_image(image):
    """Normalize image to range [0, 1]."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def remove_noise(image):
    """Apply Gaussian blur to remove noise."""
    return filters.gaussian(image, sigma=1)

def enhance_contrast(image):
    """Enhance image contrast using histogram equalization."""
    return exposure.equalize_hist(image)

def resize_image(image, target_size=(256, 256)):
    """Resize image to target size."""
    return resize(image, target_size, anti_aliasing=True)

def preprocess_image(image_path):
    """Complete preprocessing pipeline for a single image."""
    # Load image
    image = load_image(image_path)
    
    # Convert to grayscale
    image = convert_to_grayscale(image)
    
    # Normalize
    image = normalize_image(image)
    
    # Remove noise
    image = remove_noise(image)
    
    # Enhance contrast
    image = enhance_contrast(image)
    
    # Resize
    image = resize_image(image)
    
    return image 