"""
Utility functions for visualization and preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import config


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels (one-hot encoded or integers)
        y_pred: Predicted labels (one-hot encoded or integers)
        save_path: Optional path to save the plot
    """
    # Convert one-hot to integers if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get gesture labels
    labels = [config.GESTURE_LABELS[i] for i in range(config.NUM_CLASSES)]
    
    # Plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Hand Gesture Recognition', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels (one-hot encoded or integers)
        y_pred: Predicted labels (one-hot encoded or integers)
    """
    # Convert one-hot to integers if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Get gesture labels
    labels = [config.GESTURE_LABELS[i] for i in range(config.NUM_CLASSES)]
    
    print("\nClassification Report:")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=labels))


def visualize_samples(X_data, y_data, num_samples=25):
    """
    Visualize random samples from the dataset
    
    Args:
        X_data: Image data
        y_data: Labels (one-hot encoded or integers)
        num_samples: Number of samples to display
    """
    # Convert one-hot to integers if needed
    if len(y_data.shape) > 1:
        y_labels = np.argmax(y_data, axis=1)
    else:
        y_labels = y_data
    
    # Random indices
    indices = np.random.choice(len(X_data), num_samples, replace=False)
    
    # Plot
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < num_samples:
            i = indices[idx]
            img = X_data[i].squeeze()
            label = config.GESTURE_LABELS[y_labels[i]]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {label}', fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def preprocess_frame(frame):
    """
    Preprocess camera frame for model prediction
    
    Args:
        frame: Input frame (ROI) from camera
        
    Returns:
        Preprocessed frame ready for model input
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    resized = cv2.resize(gray, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input (1, 28, 28, 1)
    processed = normalized.reshape(1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    
    return processed


def label_to_gesture(label_index):
    """
    Convert label index to gesture name
    
    Args:
        label_index: Integer label (0-23)
        
    Returns:
        Gesture name (A-Y)
    """
    return config.GESTURE_LABELS.get(label_index, "Unknown")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print(f"Available gesture labels: {list(config.GESTURE_LABELS.values())}")
