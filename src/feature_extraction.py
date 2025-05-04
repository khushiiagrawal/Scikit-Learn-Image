import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.measure import regionprops
from skimage.filters import sobel

def extract_hog_features(image):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                  cells_per_block=(1, 1), visualize=False)
    return features

def extract_lbp_features(image):
    """Extract LBP (Local Binary Pattern) features."""
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    return hist

def extract_texture_features(image):
    """Extract texture features using Sobel operator."""
    edges = sobel(image)
    return np.mean(edges), np.std(edges), np.max(edges)

def extract_intensity_features(image):
    """Extract intensity-based features."""
    return np.mean(image), np.std(image), np.median(image)

def extract_all_features(image):
    """Extract all features from the image."""
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    texture_features = extract_texture_features(image)
    intensity_features = extract_intensity_features(image)
    
    # Combine all features
    all_features = np.concatenate([
        hog_features,
        lbp_features,
        texture_features,
        intensity_features
    ])
    
    return all_features 