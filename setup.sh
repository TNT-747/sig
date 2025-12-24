#!/bin/bash

echo "========================================"
echo "Hand Gesture Recognition - Setup Script"
echo "========================================"

# Create virtual environment (optional but recommended)
echo ""
echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[4/4] Installing dependencies..."
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn seaborn pillow

echo ""
echo "========================================"
echo "✓ Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Train the model: python train.py"
echo "  3. Test with camera: python predict.py"
echo ""
