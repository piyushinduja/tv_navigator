"""
Gesture to Text Application

Real-time gesture recognition that displays the recognized text phrase.

Usage:
    python app.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extract_landmarks(hand_landmarks):
    """Extract and normalize hand landmark coordinates (same as data_collector.py)"""
    landmarks = []
    
    # Get wrist coordinates for normalization
    wrist = hand_landmarks.landmark[0]
    
    # Extract all 21 landmarks (x, y, z coordinates)
    for landmark in hand_landmarks.landmark:
        # Normalize relative to wrist
        landmarks.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])
    
    return landmarks

def main():
    model_file = "gesture_classifier.pkl"
    
    # Check if model exists
    if not os.path.exists(model_file):
        print(f"✗ Error: {model_file} not found!")
        print("  Please run train_model.py first to train the classifier.")
        return
    
    # Load model
    print("Loading gesture classifier...")
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    print(f"✓ Model loaded! Recognized gestures: {list(clf.classes_)}")
    
    cap = cv2.VideoCapture(0)
    
    print("\n" + "=" * 60)
    print("GESTURE TO TEXT APPLICATION")
    print("=" * 60)
    print("Show your hand and perform gestures!")
    print("Press 'Q' to quit")
    print("=" * 60 + "\n")
    
    # Smoothing predictions with a buffer
    prediction_buffer = deque(maxlen=5)
    current_text = ""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            continue
        
        # Flip frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and predict
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Extract landmarks and predict
                landmarks = extract_landmarks(hand_landmarks)
                prediction = clf.predict([landmarks])[0]
                confidence = clf.predict_proba([landmarks]).max()
                
                # Buffer predictions for smoothing
                prediction_buffer.append(prediction)
                
                # Use most common prediction in buffer
                if len(prediction_buffer) >= 3:
                    from collections import Counter
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    current_text = most_common
        else:
            # No hand detected
            prediction_buffer.clear()
            current_text = ""
        
        # Display the recognized text
        if current_text:
            # Large text overlay
            text_size = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 60
            
            # Background rectangle for better visibility
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, current_text, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 255, 0), 3)
        else:
            # Show instruction when no hand
            cv2.putText(frame, "Show your hand to start", 
                       (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 165, 255), 2)
        
        # Show info
        cv2.putText(frame, "Press 'Q' to quit", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        cv2.imshow('Gesture to Text', frame)
        
        # Handle keyboard input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓ Application closed")

if __name__ == "__main__":
    main()
