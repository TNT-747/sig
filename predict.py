"""
Real-time hand gesture prediction using webcam
"""

import cv2
import numpy as np
import argparse
import os
import config
from model import load_trained_model
from hand_detector import HandDetector
from utils import preprocess_frame, label_to_gesture


class GesturePredictior:
    """
    Real-time gesture prediction class
    """
    
    def __init__(self, model_path=None):
        """
        Initialize gesture predictor
        
        Args:
            model_path: Path to trained model (default: best model)
        """
        # Load model
        if model_path is None:
            model_path = config.BEST_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train the model first using 'python train.py'"
            )
        
        print(f"Loading model from {model_path}...")
        self.model = load_trained_model(model_path)
        
        # Initialize hand detector
        self.hand_detector = HandDetector()
        
        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 5
        
    def predict_gesture(self, frame):
        """
        Predict gesture from frame
        
        Args:
            frame: Input frame (ROI)
            
        Returns:
            predicted_gesture: Predicted gesture letter
            confidence: Prediction confidence
            probabilities: All class probabilities
        """
        # Preprocess frame
        processed = preprocess_frame(frame)
        
        # Predict
        probabilities = self.model.predict(processed, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Convert to gesture
        predicted_gesture = label_to_gesture(predicted_class)
        
        # Smooth predictions using history
        self.prediction_history.append(predicted_class)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Get most common prediction
        if len(self.prediction_history) == self.history_size:
            smoothed_class = max(set(self.prediction_history), 
                               key=self.prediction_history.count)
            smoothed_gesture = label_to_gesture(smoothed_class)
            smoothed_confidence = probabilities[smoothed_class]
        else:
            smoothed_gesture = predicted_gesture
            smoothed_confidence = confidence
        
        return smoothed_gesture, smoothed_confidence, probabilities
    
    def display_prediction(self, frame, gesture, confidence, probabilities, 
                          show_top_k=3):
        """
        Display prediction on frame
        
        Args:
            frame: Input frame
            gesture: Predicted gesture
            confidence: Prediction confidence
            probabilities: All class probabilities
            show_top_k: Number of top predictions to show
        """
        height, width = frame.shape[:2]
        
        # Create a semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Main prediction
        text = f"Gesture: {gesture}"
        cv2.putText(
            frame, text,
            (20, 50),
            config.FONT,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            config.FONT_THICKNESS
        )
        
        # Confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        conf_color = (0, 255, 0) if confidence > config.CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(
            frame, conf_text,
            (20, 90),
            config.FONT,
            0.7,
            conf_color,
            2
        )
        
        # Show top-k predictions
        top_k_indices = np.argsort(probabilities)[-show_top_k:][::-1]
        y_offset = 130
        
        for i, idx in enumerate(top_k_indices):
            gesture_label = label_to_gesture(idx)
            prob = probabilities[idx]
            top_text = f"{i+1}. {gesture_label}: {prob*100:.1f}%"
            cv2.putText(
                frame, top_text,
                (20, y_offset + i*30),
                config.FONT,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame


def run_prediction(model_path=None, camera_id=None):
    """
    Run real-time gesture prediction
    
    Args:
        model_path: Path to trained model
        camera_id: Camera ID (default: 0)
    """
    if camera_id is None:
        camera_id = config.CAMERA_ID
    
    print("=" * 80)
    print("HAND GESTURE RECOGNITION - REAL-TIME PREDICTION")
    print("=" * 80)
    
    # Initialize predictor
    try:
        predictor = GesturePredictior(model_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Initialize camera
    print(f"\nInitializing camera (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. Camera permissions are granted")
        print("  3. No other application is using the camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    print("✓ Camera initialized successfully!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset background")
    print("  's' - Save current frame")
    print("\nPlease wait for background calibration...")
    print("Keep your hand OUT of the green box for background learning.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Draw ROI box
        frame = predictor.hand_detector.draw_roi_box(frame)
        
        # Get ROI
        roi = predictor.hand_detector.get_roi(frame)
        
        # Check if background is learned
        if not predictor.hand_detector.is_background_learned():
            # Still learning background
            predictor.hand_detector.segment_hand(roi)
            
            # Show calibration status
            remaining = predictor.hand_detector.learning_frames - predictor.hand_detector.num_frames
            status_text = f"Calibrating background... {remaining} frames"
            cv2.putText(
                frame, status_text,
                (20, frame.shape[0] - 30),
                config.FONT,
                0.7,
                (0, 255, 255),
                2
            )
        else:
            # Background learned, start prediction
            if frame_count == predictor.hand_detector.learning_frames:
                print("\n✓ Background calibration complete!")
                print("You can now place your hand in the green box.")
            
            # Segment hand
            mask = predictor.hand_detector.segment_hand(roi)
            
            if mask is not None:
                # Check if hand is present (sufficient white pixels)
                hand_present = np.sum(mask == 255) > 1000
                
                if hand_present:
                    # Predict gesture
                    gesture, confidence, probabilities = predictor.predict_gesture(roi)
                    
                    # Display prediction
                    frame = predictor.display_prediction(
                        frame, gesture, confidence, probabilities
                    )
                    
                    # Show segmented hand (for debugging)
                    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    combined_roi = np.hstack([roi, mask_3channel])
                    cv2.imshow('Hand Segmentation', combined_roi)
                else:
                    # No hand detected
                    cv2.putText(
                        frame, "No hand detected",
                        (20, 50),
                        config.FONT,
                        0.8,
                        (0, 0, 255),
                        2
                    )
        
        frame_count += 1
        
        # Display main frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            print("\nResetting background...")
            predictor.hand_detector.reset_background()
            frame_count = 0
        elif key == ord('s'):
            # Save current frame
            save_path = f"gesture_capture_{frame_count}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"Frame saved to {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Prediction session ended.")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Real-time hand gesture prediction')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    run_prediction(model_path=args.model, camera_id=args.camera)


if __name__ == "__main__":
    main()
