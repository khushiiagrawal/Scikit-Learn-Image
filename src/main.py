import os
import numpy as np
from preprocessing import preprocess_image
from feature_extraction import extract_all_features
from model import BrainTumorClassifier
import matplotlib.pyplot as plt

def load_dataset(data_dir):
    """Load and preprocess all images in the given directory."""
    X = []
    y = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    # Check if there are any class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {data_dir}")
    
    # Assuming data is organized in subdirectories by class
    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Check if there are any images in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
            
        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            
            try:
                # Preprocess image
                preprocessed_image = preprocess_image(image_path)
                
                # Extract features
                features = extract_all_features(preprocessed_image)
                
                X.append(features)
                y.append(1 if class_name == 'tumor' else 0)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
    
    if not X:
        raise ValueError(f"No valid images found in {data_dir}")
    
    return np.array(X), np.array(y)

def main():
    try:
        # Create necessary directories
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/test', exist_ok=True)
        os.makedirs('data/validation', exist_ok=True)
        
        # Load and preprocess training data
        print("Loading and preprocessing training data...")
        X_train, y_train = load_dataset('data/train')
        
        # Initialize and train the model
        print("Training the model...")
        classifier = BrainTumorClassifier()
        accuracy, report = classifier.train(X_train, y_train)
        
        print(f"Validation Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)
        
        # Save the trained model
        classifier.save_model('brain_tumor_classifier.joblib')
        print("\nModel saved as 'brain_tumor_classifier.joblib'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. Created the directory structure:")
        print("   data/train/tumor/")
        print("   data/train/no_tumor/")
        print("2. Added MRI brain images to these directories")
        print("3. Images should be in PNG or JPG format")

if __name__ == "__main__":
    main() 