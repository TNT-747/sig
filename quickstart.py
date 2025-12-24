"""
Quick Start Guide for Hand Gesture Recognition System
Test the system components without full training
"""

import os
import sys

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("TESTING PACKAGE IMPORTS")
    print("=" * 60)
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow'
    }
    
    failed = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name:15s} - OK")
        except ImportError as e:
            print(f"✗ {name:15s} - FAILED: {e}")
            failed.append(name)
    
    print("=" * 60)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        print("  OR run: bash setup.sh")
        return False
    else:
        print("\n✓ All packages installed successfully!")
        return True


def test_dataset():
    """Test if dataset files exist and are readable"""
    print("\n" + "=" * 60)
    print("TESTING DATASET")
    print("=" * 60)
    
    import pandas as pd
    
    train_file = 'sign_mnist_train.csv'
    test_file = 'sign_mnist_test.csv'
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        return False
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"✓ Found: {train_file}")
    print(f"✓ Found: {test_file}")
    
    # Read first few rows
    print("\nReading dataset...")
    try:
        train_df = pd.read_csv(train_file, nrows=100)
        test_df = pd.read_csv(test_file, nrows=100)
        
        print(f"✓ Training data shape (sample): {train_df.shape}")
        print(f"✓ Test data shape (sample): {test_df.shape}")
        print(f"✓ Features: {train_df.shape[1] - 1} (pixels)")
        print(f"✓ Unique labels: {sorted(train_df['label'].unique())}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


def test_camera():
    """Test if camera is accessible"""
    print("\n" + "=" * 60)
    print("TESTING CAMERA")
    print("=" * 60)
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            print("\nTroubleshooting:")
            print("  1. Check if camera is connected")
            print("  2. Check camera permissions")
            print("  3. Try a different camera ID (--camera 1)")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Could not read from camera")
            return False
        
        print(f"✓ Camera is accessible")
        print(f"✓ Frame shape: {frame.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False


def test_config():
    """Test configuration file"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        import config
        
        print(f"✓ Config loaded successfully")
        print(f"  - Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
        print(f"  - Number of classes: {config.NUM_CLASSES}")
        print(f"  - Batch size: {config.BATCH_SIZE}")
        print(f"  - Epochs: {config.EPOCHS}")
        print(f"  - Learning rate: {config.LEARNING_RATE}")
        print(f"  - Model path: {config.MODEL_PATH}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "HAND GESTURE RECOGNITION - QUICK START" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {
        'Packages': test_imports(),
        'Dataset': test_dataset(),
        'Configuration': test_config(),
        'Camera': test_camera()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Train the model:")
        print("     python train.py")
        print("\n  2. After training, test with your camera:")
        print("     python predict.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing dataset: Download Sign Language MNIST CSV files")
        print("  - Camera issues: Check permissions and connections")
    
    print()


if __name__ == "__main__":
    main()
