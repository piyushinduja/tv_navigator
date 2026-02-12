"""
Data Collection Tool for Gesture-to-Text Training

This script allows you to collect hand gesture data and associate it with text labels.
Press 'R' to record samples, 'Q' to quit, 'N' to enter a new gesture label.

Usage:
    python data_collector.py
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Data storage
csv_file = "gesture_data.csv"
current_label = None
sample_count = 0

def extract_landmarks(hand_landmarks):
    """Extract and normalize hand landmark coordinates"""
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

def save_sample(landmarks, label):
    """Save a sample to the CSV file"""
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = []
            for i in range(21):
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            header.append('label')
            writer.writerow(header)
        
        # Write landmark data + label
        row = landmarks + [label]
        writer.writerow(row)

def main():
    global current_label, sample_count
    
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("GESTURE DATA COLLECTION TOOL")
    print("=" * 60)
    print("\nControls:")
    print("  N - Enter new gesture label (text phrase)")
    print("  R - Record sample (hold to record multiple)")
    print("  Q - Quit and close")
    print("\nMake sure your hand is visible in the camera!")
    print("=" * 60)
    
    recording = False
    
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
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # If recording and we have a label, save the sample
                if recording and current_label:
                    landmarks = extract_landmarks(hand_landmarks)
                    save_sample(landmarks, current_label)
                    sample_count += 1
                    
                    # Visual feedback
                    cv2.putText(frame, f"RECORDING! Samples: {sample_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
        
        # Display current label
        if current_label:
            cv2.putText(frame, f"Label: {current_label}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press 'N' to set label", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 165, 255), 2)
        
        cv2.putText(frame, f"Total Samples: {sample_count}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Data Collection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"\n✓ Session complete! Collected {sample_count} samples total.")
            break
        elif key == ord('n'):
            # Get new label from user
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            new_label = input("Enter text phrase for this gesture: ").strip()
            if new_label:
                current_label = new_label
                print(f"✓ Label set to: '{current_label}'")
                print("  Now hold 'R' to record samples for this gesture")
            else:
                print("✗ Invalid label. Try again.")
            print("=" * 60 + "\n")
            
            cap = cv2.VideoCapture(0)
            
        elif key == ord('r'):
            if current_label:
                recording = True
            else:
                print("Please set a label first (press 'N')")
        else:
            recording = False
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if sample_count > 0:
        print(f"\n✓ Data saved to: {csv_file}")
        print(f"✓ You can now run train_model.py to train your classifier!")
    else:
        print("\n✗ No samples collected.")

if __name__ == "__main__":
    main()
