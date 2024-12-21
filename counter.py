import cv2
import numpy as np
from ultralytics import YOLO  # We'll use YOLOv8 for person detection

class PeopleCounter:
    def __init__(self, video_path, line_position=0.4):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO('yolov8n.pt')  # Using the smallest YOLOv8 model
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define crossing line (70% of frame width by default)
        self.line_position = int(self.frame_width * line_position)
        
        # Counters
        self.entries = 0
        self.exits = 0
        
        # Track previous positions
        self.previous_positions = {}
        
    def process_frame(self, frame):
        # Run YOLOv8 detection
        results = self.model(frame, classes=0)  # class 0 is person
        
        # Draw crossing line
        cv2.line(frame, (self.line_position, 0), (self.line_position, self.frame_height), 
                 (0, 255, 0), 2)
        
        current_positions = {}
        
        # Process each detection
        for r in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = r
            if conf > 0.3:  # Confidence threshold
                # Calculate center point of the bounding box
                center_x = (x1 + x2) / 2
                
                # Store current position
                box_id = len(current_positions)
                current_positions[box_id] = center_x
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Check crossing
                if box_id in self.previous_positions:
                    prev_x = self.previous_positions[box_id]
                    
                    # Detect crossing from right to left (exit)
                    if prev_x > self.line_position and center_x < self.line_position:
                        self.exits += 1
                    
                    # Detect crossing from left to right (entry)
                    elif prev_x < self.line_position and center_x > self.line_position:
                        self.entries += 1
        
        # Update previous positions
        self.previous_positions = current_positions
        
        # Draw counters on frame
        cv2.putText(frame, f"giris: {self.entries}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"cikis: {self.exits}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', 
                            fourcc, 
                            30.0,  # FPS
                            (self.frame_width, self.frame_height))
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Write the frame to output video
            out.write(processed_frame)
            
            # Display the frame (optional - you can comment this out)
            cv2.imshow('People Counter', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release everything
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/demo/video.mp4"
    counter = PeopleCounter(video_path)
    counter.run() 