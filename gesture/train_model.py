"""
Train Gesture Classifier

This script trains a Random Forest classifier on the collected gesture data.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os

def train_model():
    csv_file = "gesture_data.csv"
    
    # Check if data file exists
    if not os.path.exists(csv_file):
        print(f"✗ Error: {csv_file} not found!")
        print("  Please run data_collector.py first to collect training data.")
        return
    
    print("=" * 60)
    print("TRAINING GESTURE CLASSIFIER")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv(csv_file)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Check label distribution
    print(f"\n[2/5] Data distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  - '{label}': {count} samples")
    
    if len(df) < 10:
        print("\n✗ Warning: Very few samples! Collect at least 50+ samples per gesture for best results.")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Split data
    print(f"\n[3/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Training samples: {len(X_train)}")
    print(f"  ✓ Test samples: {len(X_test)}")
    
    # Train model
    print(f"\n[4/5] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    # Evaluate
    print(f"\n[5/5] Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✓ Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_file = "gesture_classifier.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"✓ Model saved to: {model_file}")
    
    # Save label mapping for reference
    label_mapping = {i: label for i, label in enumerate(clf.classes_)}
    with open("label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✓ Label mapping saved to: label_mapping.json")
    
    print("\n" + "=" * 60)
    print("✓ Training complete! You can now run app.py")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
