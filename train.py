"""
Training pipeline for hand gesture recognition model
"""

import os
import argparse
from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
from data_loader import load_data, create_data_generators
from model import create_model
from utils import (
    plot_training_history, 
    plot_confusion_matrix, 
    print_classification_report
)


def train_model(epochs=None, batch_size=None):
    """
    Train the hand gesture recognition model
    
    Args:
        epochs: Number of training epochs (default: from config)
        batch_size: Batch size (default: from config)
        
    Returns:
        Trained model and training history
    """
    # Use config values if not specified
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    print("=" * 80)
    print("HAND GESTURE RECOGNITION - TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # Create data generators
    print("\n[2/5] Creating data generators with augmentation...")
    train_generator, val_generator = create_data_generators(X_train, y_train)
    
    # Create model
    print("\n[3/5] Building CNN model...")
    model = create_model()
    model.summary()
    
    # Setup callbacks
    print("\n[4/5] Setting up training callbacks...")
    
    # Model checkpoint - save best model
    checkpoint = ModelCheckpoint(
        config.BEST_MODEL_PATH,
        monitor=config.MONITOR_METRIC,
        save_best_only=config.SAVE_BEST_ONLY,
        mode='max',
        verbose=1
    )
    
    # Early stopping - prevent overfitting
    early_stop = EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Train model
    print("\n[5/5] Training model...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("-" * 80)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = config.MODEL_PATH
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    print(f"✓ Best model saved to: {config.BEST_MODEL_PATH}")
    
    return model, history, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test set
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*80}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*80}")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    return test_accuracy


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train hand gesture recognition model')
    parser.add_argument('--epochs', type=int, default=None, 
                       help=f'Number of epochs (default: {config.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {config.BATCH_SIZE})')
    
    args = parser.parse_args()
    
    # Train model
    start_time = datetime.now()
    model, history, X_test, y_test = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    end_time = datetime.now()
    
    # Plot training history
    print("\nGenerating training history plots...")
    history_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Training summary
    training_time = end_time - start_time
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Training time: {training_time}")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("=" * 80)
    print("\n✓ Training completed successfully!")
    print("\nNext steps:")
    print("  1. Review the confusion matrix and training history plots")
    print("  2. Run 'python predict.py' to test with your webcam")


if __name__ == "__main__":
    main()
