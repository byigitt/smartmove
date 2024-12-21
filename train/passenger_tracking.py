"""
Full-body human detection and tracking system
"""

import cv2
import numpy as np
import logging
from datetime import datetime

class PassengerTracker:
    def __init__(self, video_source: str, line_position: float = 0.5):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
            
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Vertical counting line
        self.line_x = int(self.frame_width * line_position)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Human body detection parameters
        self.min_area = 15000  # Minimum area for a human body
        self.max_area = 100000  # Maximum area for a human body
        self.min_aspect_ratio = 1.5  # Minimum height/width ratio for human body
        self.max_aspect_ratio = 4.0  # Maximum height/width ratio for human body
        self.min_solidity = 0.5  # Minimum solidity (area/convex hull area)
        
        # Tracking parameters
        self.last_x = None
        self.direction = None
        self.tracking_history = []
        self.history_length = 5
        self.min_movement = 3  # Minimum pixels to consider as movement
        self.confidence_threshold = 3  # Number of consistent movements needed
        self.movement_buffer = []  # Buffer for movement direction
        self.buffer_size = 5
        
        # Counters
        self.entering = 0
        self.leaving = 0
        
        # Debug mode
        self.debug = True
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def is_human_shape(self, contour, x, y, w, h):
        """Enhanced human shape detection with multiple criteria"""
        # Check basic area
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False, 0
            
        # Check aspect ratio
        aspect_ratio = h / w
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, 0
            
        # Check solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            if solidity < self.min_solidity:
                return False, 0
                
        # Calculate confidence score (0-1)
        aspect_score = min(1.0, (aspect_ratio - self.min_aspect_ratio) / 
                         (self.max_aspect_ratio - self.min_aspect_ratio))
        area_score = min(1.0, (area - self.min_area) / 
                       (self.max_area - self.min_area))
        solidity_score = solidity if hull_area > 0 else 0
        
        confidence = (aspect_score + area_score + solidity_score) / 3.0
        
        return True, confidence
        
    def detect_human(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Noise removal and hole filling
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find human-shaped contours with confidence
        human_detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            is_human, confidence = self.is_human_shape(contour, x, y, w, h)
            if is_human:
                human_detections.append((x, y, w, h, confidence))
        
        # Return the detection with highest confidence
        if human_detections:
            best_detection = max(human_detections, key=lambda x: x[4])
            x, y, w, h, conf = best_detection
            center_x = x + w//2
            center_y = y + h//2
            return (x, y, w, h, conf), (center_x, center_y)
                
        return None, None
        
    def update_tracking(self, center, confidence):
        if center is None:
            self.last_x = None
            self.direction = None
            self.tracking_history.clear()
            self.movement_buffer.clear()
            return
            
        current_x = center[0]
        
        # Update tracking history
        self.tracking_history.append(current_x)
        if len(self.tracking_history) > self.history_length:
            self.tracking_history.pop(0)
        
        # Initialize tracking
        if self.last_x is None:
            self.last_x = current_x
            return
            
        # Calculate movement
        movement = current_x - self.last_x
        
        # Update movement buffer
        if abs(movement) > self.min_movement:
            direction = 'left' if movement < 0 else 'right'
            self.movement_buffer.append(direction)
            if len(self.movement_buffer) > self.buffer_size:
                self.movement_buffer.pop(0)
            
            # Check for consistent movement
            if len(self.movement_buffer) >= self.confidence_threshold:
                if all(d == 'left' for d in self.movement_buffer[-self.confidence_threshold:]):
                    self.direction = 'left'
                elif all(d == 'right' for d in self.movement_buffer[-self.confidence_threshold:]):
                    self.direction = 'right'
                
        # Check line crossing with confidence threshold
        if self.direction and confidence > 0.6:  # Only count if confidence is high enough
            if self.direction == 'left' and self.last_x > self.line_x and current_x <= self.line_x:
                self.entering += 1
                self.logger.info(f"Person entering - Total: {self.entering} (Confidence: {confidence:.2f})")
            elif self.direction == 'right' and self.last_x < self.line_x and current_x >= self.line_x:
                self.leaving += 1
                self.logger.info(f"Person leaving - Total: {self.leaving} (Confidence: {confidence:.2f})")
                
        self.last_x = current_x
        
    def draw_frame(self, frame, detection, center):
        # Draw counting line
        cv2.line(frame, (self.line_x, 0), (self.line_x, self.frame_height), 
                (0, 0, 255), 2)  # Red line
                
        # Draw detection
        if detection is not None:
            x, y, w, h, confidence = detection
            # Color based on confidence (green to red)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Draw full-body rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence and dimensions
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Size: {w}x{h}", (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if center is not None:
            # Draw center point
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            
            # Draw direction arrow
            if self.direction:
                arrow_length = 50
                arrow_x = center[0]
                arrow_y = center[1]
                if self.direction == 'left':
                    end_x = arrow_x - arrow_length
                    color = (0, 255, 0)  # Green for entering
                else:
                    end_x = arrow_x + arrow_length
                    color = (0, 0, 255)  # Red for leaving
                cv2.arrowedLine(frame, center, (end_x, arrow_y), color, 2)
        
        # Draw counts
        cv2.putText(frame, f"Entering: {self.entering}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Leaving: {self.leaving}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                   
        return frame
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect human
            detection, center = self.detect_human(frame)
            
            # Update tracking
            confidence = detection[4] if detection else 0
            self.update_tracking(center, confidence)
            
            # Draw visualization
            frame = self.draw_frame(frame, detection, center)
            
            # Display
            cv2.imshow('Human Body Tracking', frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Log final counts
        self.logger.info(f"Final counts - Entering: {self.entering}, Leaving: {self.leaving}")
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Human Body Detection and Tracking')
    parser.add_argument('video_path', help='Path to video file or camera index')
    parser.add_argument('--line-pos', type=float, default=0.5,
                      help='Position of counting line (0-1)')
    
    args = parser.parse_args()
    
    tracker = PassengerTracker(args.video_path, args.line_pos)
    tracker.run()

if __name__ == '__main__':
    main() 