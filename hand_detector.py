"""
Hand detector using OpenCV for gesture recognition
Implements background subtraction for hand ROI extraction
"""

import cv2
import numpy as np
import config


class HandDetector:
    """
    Hand detector class using background subtraction and skin detection
    """
    
    def __init__(self):
        """Initialize hand detector"""
        self.bg_subtractor = None
        self.num_frames = 0
        self.learning_frames = 30  # Number of frames to learn background
        
    def get_roi(self, frame):
        """
        Extract Region of Interest (ROI) from frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            ROI region for hand detection
        """
        # Define ROI coordinates
        roi = frame[
            config.ROI_TOP:config.ROI_BOTTOM,
            config.ROI_LEFT:config.ROI_RIGHT
        ]
        return roi
    
    def draw_roi_box(self, frame):
        """
        Draw ROI bounding box on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with ROI box drawn
        """
        cv2.rectangle(
            frame,
            (config.ROI_LEFT, config.ROI_TOP),
            (config.ROI_RIGHT, config.ROI_BOTTOM),
            config.BOX_COLOR,
            2
        )
        
        # Add instruction text
        text = "Place your hand in the green box"
        cv2.putText(
            frame,
            text,
            (config.ROI_LEFT, config.ROI_TOP - 10),
            config.FONT,
            0.6,
            config.TEXT_COLOR,
            1
        )
        
        return frame
    
    def segment_hand(self, frame):
        """
        Segment hand from background using background subtraction
        
        Args:
            frame: Input frame (ROI)
            
        Returns:
            Segmented hand region (binary mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Initialize background subtractor if needed
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                detectShadows=False
            )
        
        # Apply background subtraction if background is learned
        if self.num_frames < self.learning_frames:
            # Learning phase - just update background
            self.bg_subtractor.apply(blurred, learningRate=0.5)
            self.num_frames += 1
            return None
        else:
            # Detection phase
            mask = self.bg_subtractor.apply(blurred, learningRate=0)
            
            # Apply threshold
            _, thresholded = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            
            return thresholded
    
    def is_background_learned(self):
        """
        Check if background has been learned
        
        Returns:
            True if background is learned, False otherwise
        """
        return self.num_frames >= self.learning_frames
    
    def reset_background(self):
        """Reset background subtractor"""
        self.bg_subtractor = None
        self.num_frames = 0
        print("Background reset. Please remove your hand from the ROI.")


def apply_skin_detection(frame):
    """
    Apply skin color detection as an alternative method
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Binary mask of skin regions
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    return mask


if __name__ == "__main__":
    print("Hand detector module loaded successfully!")
    print(f"ROI coordinates: ({config.ROI_LEFT}, {config.ROI_TOP}) to ({config.ROI_RIGHT}, {config.ROI_BOTTOM})")
