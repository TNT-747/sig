"""
CNN Model architecture for hand gesture recognition
"""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import config


def create_model():
    """
    Create CNN model for hand gesture recognition
    
    Architecture:
        - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool + Dropout)
        - Flatten layer
        - Dense layer with dropout
        - Output layer with softmax
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(
            config.CNN_FILTERS[0], 
            config.KERNEL_SIZE, 
            activation='relu',
            input_shape=config.INPUT_SHAPE,
            padding='same'
        ),
        BatchNormalization(),
        Conv2D(
            config.CNN_FILTERS[0], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D(config.POOL_SIZE),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(
            config.CNN_FILTERS[1], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        Conv2D(
            config.CNN_FILTERS[1], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D(config.POOL_SIZE),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(
            config.CNN_FILTERS[2], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D(config.POOL_SIZE),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(config.DENSE_UNITS, activation='relu'),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        Dense(config.DENSE_UNITS // 2, activation='relu'),
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),
        
        # Output Layer
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_trained_model(model_path=None):
    """
    Load a pre-trained model
    
    Args:
        model_path: Path to saved model (default: config.BEST_MODEL_PATH)
        
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    
    print(f"Loading model from {model_path}...")
    try:
        # Try loading with safe_mode=False to ignore unknown parameters
        model = load_model(model_path, safe_mode=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model with safe_mode=False: {e}")
        print("Trying to load with default settings...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    model = create_model()
    
    print("\nModel Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
