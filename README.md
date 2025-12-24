# Hand Gesture Recognition System 🤚

A real-time hand gesture recognition system using Deep Learning (CNN) and Computer Vision (OpenCV). The system recognizes 24 different hand gestures (A-Y in American Sign Language, excluding J and Z which require motion).

## 🎯 Project Overview

This project implements a complete pipeline for hand gesture recognition:
- **Data Loading & Preprocessing**: Uses Sign Language MNIST dataset
- **Model Architecture**: Convolutional Neural Network (CNN) with batch normalization and dropout
- **Training Pipeline**: Includes data augmentation, model checkpointing, and early stopping
- **Real-time Prediction**: Webcam-based gesture recognition with background subtraction

## 📊 Dataset

**Sign Language MNIST Dataset**
- Training samples: 27,455 images
- Test samples: 7,172 images
- Image size: 28x28 pixels (grayscale)
- Classes: 24 gestures (A-Y, excluding J and Z)

The dataset files are:
- `sign_mnist_train.csv` - Training data
- `sign_mnist_test.csv` - Test data

## 🏗️ Project Structure

```
Sig/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Data loading and preprocessing
├── model.py              # CNN model architecture
├── train.py              # Training pipeline
├── hand_detector.py      # Hand detection using OpenCV
├── predict.py            # Real-time prediction with webcam
├── utils.py              # Utility functions and visualization
├── requirements.txt      # Python dependencies
├── models/               # Saved models directory
│   ├── hand_gesture_model.h5
│   └── best_model.h5
└── results/              # Training results and visualizations
    ├── training_history.png
    └── confusion_matrix.png
```

## 🚀 Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Dependencies include:
- TensorFlow 2.15.0
- OpenCV 4.8.1
- NumPy, Pandas, Matplotlib, Scikit-learn, Seaborn

## 📖 Usage

### 1. Train the Model

Train the CNN model on the Sign Language MNIST dataset:

```bash
python train.py
```

**Optional arguments:**
- `--epochs <num>`: Number of training epochs (default: 30)
- `--batch-size <size>`: Batch size (default: 64)

Example:
```bash
python train.py --epochs 40 --batch-size 128
```

**Training Features:**
- Data augmentation (rotation, zoom, shift)
- Model checkpointing (saves best model)
- Early stopping (patience: 7 epochs)
- Learning rate reduction on plateau
- Validation split: 15%

**Expected Output:**
- Test accuracy: >90%
- Training time: ~15-30 minutes (depending on hardware)
- Saved models in `models/` directory
- Visualization plots in `results/` directory

### 2. Real-time Gesture Prediction

Use your webcam to test the trained model with your own hand gestures:

```bash
python predict.py
```

**Optional arguments:**
- `--model <path>`: Path to trained model (default: best_model.h5)
- `--camera <id>`: Camera ID (default: 0)

Example:
```bash
python predict.py --camera 1
```

**How to Use:**
1. **Background Calibration**: When you start, keep your hand OUT of the green box for ~30 frames to calibrate the background
2. **Gesture Recognition**: Place your hand inside the green box and make gestures
3. **View Results**: The predicted gesture and confidence will be displayed on screen

**Keyboard Controls:**
- `q` - Quit the application
- `r` - Reset background (recalibrate)
- `s` - Save current frame

**Tips for Better Recognition:**
- Ensure good lighting conditions
- Use a simple, contrasting background
- Keep your hand centered in the ROI box
- Make clear, distinct gestures
- Wait for background calibration to complete

### 3. Test Individual Modules

**Test data loading:**
```bash
python data_loader.py
```

**Test model creation:**
```bash
python model.py
```

**Test camera:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); cap.release(); print('Camera OK' if ret else 'Camera FAILED')"
```

## 🧠 Model Architecture

**CNN Architecture:**
- **Input**: 28x28x1 grayscale images
- **Convolutional Block 1**: 2x Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
- **Convolutional Block 2**: 2x Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
- **Convolutional Block 3**: Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
- **Fully Connected**: Dense(128) + BatchNorm + Dropout(0.5)
- **Fully Connected**: Dense(64) + BatchNorm + Dropout(0.5)
- **Output**: Dense(24, softmax)

**Total Parameters**: ~300K parameters

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Metrics: Accuracy

## 📈 Results

After training, you'll find:

1. **Training History Plot** (`results/training_history.png`)
   - Shows accuracy and loss curves for training and validation

2. **Confusion Matrix** (`results/confusion_matrix.png`)
   - Displays per-class performance

3. **Classification Report**
   - Precision, recall, and F1-score for each gesture

## 🎓 Educational Objectives

This project demonstrates:

✅ **Data Collection & Annotation**
- Working with CSV-based image datasets
- Data preprocessing and normalization

✅ **Computer Vision with OpenCV**
- Camera capture and frame processing
- Background subtraction for hand segmentation
- ROI extraction and preprocessing

✅ **Deep Learning**
- Building CNN architectures
- Training with data augmentation
- Model evaluation and interpretation

✅ **Real-time Applications**
- Integrating ML models with webcam input
- Real-time prediction and visualization

## 🛠️ Configuration

All configurations are in `config.py`:

**Key Parameters:**
- `BATCH_SIZE = 64`
- `EPOCHS = 30`
- `LEARNING_RATE = 0.001`
- `VALIDATION_SPLIT = 0.15`
- `CONFIDENCE_THRESHOLD = 0.6`

**ROI Settings** (for webcam):
- Default ROI: (300, 100) to (600, 400)
- Adjustable in `config.py` if needed

## 🔧 Troubleshooting

**Issue: Camera not working**
- Check camera permissions
- Try different camera ID: `python predict.py --camera 1`
- Ensure no other app is using the camera

**Issue: Low prediction accuracy**
- Recalibrate background (press 'r')
- Improve lighting conditions
- Use simpler background
- Retrain model with more epochs

**Issue: Model not found error**
- Run training first: `python train.py`
- Check that `models/best_model.h5` exists

## 📚 References

- [Sign Language MNIST Dataset](https://www.kaggle.com/datamunge/sign-language-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try CNN+LSTM for temporal modeling
- Add more gestures
- Improve hand detection algorithms

## 📝 License

This project is for educational purposes.

---

**Author**: Built for learning hand gesture recognition with Computer Vision and Deep Learning

**Last Updated**: December 2025
# sig
