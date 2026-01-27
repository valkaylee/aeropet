#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for Hand Gesture Classifier
Trains a model using collected keypoint data, compatible with kinivi's workflow

Usage:
    python train_gesture_classifier.py
    
    Or with custom paths:
    python train_gesture_classifier.py --data model/keypoint_classifier/keypoint.csv --output model/keypoint_classifier/keypoint_classifier.tflite
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_training_data(csv_path: str):
    """
    Load training data from CSV file.
    
    Expected format: label, x0, y0, x1, y1, ..., x20, y20
    Where label is 0-9 and there are 42 features (21 landmarks * 2 coordinates)
    
    Returns:
        X: numpy array of shape (n_samples, 42) - preprocessed landmarks
        y: numpy array of shape (n_samples,) - labels
    """
    X = []
    y = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data file not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 43:  # label + 42 features
                continue
            
            label = int(row[0])
            features = [float(x) for x in row[1:43]]  # 42 features
            
            X.append(features)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} samples")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def create_model(num_classes: int, input_dim: int = 42):
    """
    Create a simple MLP model for gesture classification.
    
    This matches the architecture typically used in kinivi's training.
    
    Args:
        num_classes: Number of gesture classes
        input_dim: Input feature dimension (42 for 21 landmarks * 2 coordinates)
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, y, output_path: str, epochs: int = 100, batch_size: int = 32, test_size: float = 0.2):
    """
    Train the gesture classifier model.
    
    Args:
        X: Training features (n_samples, 42)
        y: Training labels (n_samples,)
        output_path: Path to save the trained TFLite model
        epochs: Number of training epochs
        batch_size: Batch size for training
        test_size: Fraction of data to use for validation
    
    Returns:
        Trained model and training history
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Get number of classes
    num_classes = len(np.unique(y))
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Predictions for detailed metrics
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))
    
    # Convert to TFLite
    print(f"\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Model saved to: {output_path}")
    print(f"  Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train hand gesture classifier model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data CSV file (default: model/keypoint_classifier/keypoint.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trained TFLite model (default: model/keypoint_classifier/keypoint_classifier.tflite)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )
    args = parser.parse_args()
    
    # Set default paths
    if args.data is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.data = os.path.join(current_dir, 'model', 'keypoint_classifier', 'keypoint.csv')
    
    if args.output is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(current_dir, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
    
    print("="*60)
    print("Hand Gesture Classifier Training")
    print("="*60)
    print(f"Training data: {args.data}")
    print(f"Output model: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Load training data
    print("\nLoading training data...")
    try:
        X, y = load_training_data(args.data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease collect training data first using:")
        print("  python collect_training_data.py --camera 0")
        return
    
    if len(X) == 0:
        print("Error: No training data found in CSV file")
        return
    
    # Check minimum samples per class
    min_samples = np.min(np.bincount(y))
    if min_samples < 10:
        print(f"\nWarning: Some classes have very few samples (minimum: {min_samples})")
        print("Consider collecting more data for better model performance")
    
    # Train model
    try:
        model, history = train_model(
            X, y,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size
        )
        
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"Model saved to: {args.output}")
        print("\nYou can now use this model with gesture_demo.py")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

