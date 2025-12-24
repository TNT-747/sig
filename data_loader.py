"""
Data loading and preprocessing module for Sign Language MNIST dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def load_data():
    """
    Load and preprocess Sign Language MNIST dataset
    
    Returns:
        X_train: Training images (N, 28, 28, 1)
        y_train: Training labels (N, 24) - one-hot encoded
        X_test: Test images (N, 28, 28, 1)
        y_test: Test labels (N, 24) - one-hot encoded
    """
    print("Loading training data...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    
    print("Loading test data...")
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    # Separate features and labels
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    
    # Reshape to image format (N, 28, 28, 1)
    X_train = X_train.reshape(-1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    X_test = X_test.reshape(-1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, config.NUM_CLASSES)
    y_test = to_categorical(y_test, config.NUM_CLASSES)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def create_data_generators(X_train, y_train):
    """
    Create data generators with augmentation for training
    
    Args:
        X_train: Training images
        y_train: Training labels
        
    Returns:
        train_generator: Augmented training data generator
        val_generator: Validation data generator
    """
    # Create ImageDataGenerator with augmentation
    train_datagen = ImageDataGenerator(**config.AUGMENTATION_PARAMS)
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator()
    
    # Split training data into train and validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, 
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)  # Stratified split
    )
    
    print(f"Training split: {X_train_split.shape}")
    print(f"Validation split: {X_val_split.shape}")
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train_split, y_train_split,
        batch_size=config.BATCH_SIZE
    )
    
    val_generator = val_datagen.flow(
        X_val_split, y_val_split,
        batch_size=config.BATCH_SIZE
    )
    
    return train_generator, val_generator


def get_class_distribution(y_data):
    """
    Get the distribution of classes in the dataset
    
    Args:
        y_data: One-hot encoded labels
        
    Returns:
        Dictionary with class counts
    """
    labels = np.argmax(y_data, axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    
    distribution = {}
    for label, count in zip(unique, counts):
        gesture = config.GESTURE_LABELS[label]
        distribution[gesture] = count
    
    return distribution


if __name__ == "__main__":
    # Test data loading
    X_train, y_train, X_test, y_test = load_data()
    
    print("\nTraining set class distribution:")
    train_dist = get_class_distribution(y_train)
    for gesture, count in sorted(train_dist.items()):
        print(f"  {gesture}: {count}")
    
    print("\nTest set class distribution:")
    test_dist = get_class_distribution(y_test)
    for gesture, count in sorted(test_dist.items()):
        print(f"  {gesture}: {count}")
    
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(X_train, y_train)
    print("Data generators created successfully!")
