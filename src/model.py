from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

class BrainTumorClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        """Train the model on the given data."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        
        return accuracy, report
    
    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for each class."""
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        """Save the trained model to disk."""
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load a trained model from disk."""
        self.model = joblib.load(path) 