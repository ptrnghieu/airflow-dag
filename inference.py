#!/usr/bin/env python3
"""
Model Inference Script
Load trained model và predict categories cho input text
"""

import os
import joblib
import re
from typing import List, Dict, Any

def load_model():
    """Load trained model artifacts"""
    model_dir = "tmp/ml_training/models"
    
    try:
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        mlb = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        
        print("Model loaded successfully")
        return model, vectorizer, mlb
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def preprocess_text(text: str) -> str:
    """Preprocess text (same as training)"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_categories(model, vectorizer, mlb, title: str, abstract: str = "") -> Dict[str, Any]:
    """Predict categories for given title and abstract"""
    
    # Combine title and abstract
    text = f"{title} {abstract}".strip()
    
    # Preprocess
    text_cleaned = preprocess_text(text)
    
    # Vectorize
    X = vectorizer.transform([text_cleaned])
    
    # Predict
    y_pred_binary = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Convert to categories
    categories = mlb.inverse_transform(y_pred_binary)
    
    # Get probabilities for predicted categories
    predicted_categories = categories[0] if len(categories) > 0 else []
    confidence_scores = {}
    
    for cat in predicted_categories:
        if cat in mlb.classes_:
            idx = list(mlb.classes_).index(cat)
            confidence_scores[cat] = float(y_pred_proba[0][idx])
    
    return {
        "input_title": title,
        "input_abstract": abstract,
        "text_cleaned": text_cleaned,
        "predicted_categories": predicted_categories,
        "confidence_scores": confidence_scores
    }

def main():
    """Main inference function"""
    print("ArXiv Category Prediction - Inference")
    print("=" * 50)
    
    # Load model
    model, vectorizer, mlb = load_model()
    if model is None:
        return
    
    print(f"Model Info:")
    print(f"  - Classes: {len(mlb.classes_)}")
    print(f"  - Available categories: {list(mlb.classes_)}")
    
    # Test samples
    test_samples = [
        {
            "title": "Deep Learning for Computer Vision",
            "abstract": "This paper presents a novel approach to computer vision using deep learning techniques for image recognition and object detection."
        },
        {
            "title": "Machine Learning Algorithms for Natural Language Processing",
            "abstract": "We propose new machine learning methods for natural language understanding and text classification tasks."
        },
        {
            "title": "Quantum Computing and Quantum Machine Learning",
            "abstract": "This work explores quantum algorithms for machine learning applications in quantum computing systems."
        },
        {
            "title": "Robotics and Autonomous Systems",
            "abstract": "We develop autonomous robotic systems using reinforcement learning and computer vision for navigation."
        },
        {
            "title": "Cryptocurrency and Blockchain Technology",
            "abstract": "This paper analyzes blockchain technology and its applications in cryptocurrency systems and smart contracts."
        }
    ]
    
    print(f"\nTesting {len(test_samples)} samples:")
    print("=" * 50)
    
    for i, sample in enumerate(test_samples):
        print(f"\nSample {i+1}:")
        print("-" * 30)
        
        # Predict
        result = predict_categories(
            model, vectorizer, mlb, 
            sample["title"], 
            sample["abstract"]
        )
        
        # Display results
        print(f"Title: {result['input_title']}")
        print(f"Abstract: {result['input_abstract'][:100]}...")
        print(f"Cleaned: {result['text_cleaned'][:100]}...")
        print(f"Predicted Categories: {result['predicted_categories']}")
        
        if result['confidence_scores']:
            print(f"Confidence Scores:")
            for cat, score in result['confidence_scores'].items():
                print(f"   {cat}: {score:.3f}")

if __name__ == "__main__":
    main()
