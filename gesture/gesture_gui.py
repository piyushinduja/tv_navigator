"""
Gesture to Text - Unified GUI Application with Callback Support

A complete gesture recognition system with:
- Live camera feed with hand landmark tracking
- Training interface to associate gestures with text phrases
- Real-time gesture recognition and text output
- Callback support to trigger actions based on recognized gestures

Usage:
    python gesture_gui_with_callback.py
"""

# Suppress protobuf deprecation warnings from MediaPipe
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import csv
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import deque, Counter
import time

class GestureToTextGUI:
    def __init__(self, root, on_gesture_callback=None):
        """
        Initialize the GUI
        
        Args:
            root: Tkinter root window
            on_gesture_callback: Optional callback function that will be called when a gesture is recognized
                                Signature: callback(gesture_text: str, confidence: float)
        """
        self.root = root
        self.root.title("Gesture to Text - Training & Recognition")
        self.root.geometry("1200x700")
        
        # Store the callback
        self.on_gesture_callback = on_gesture_callback
        
        # MediaPipe setup - using solutions API (not tasks API)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data
        self.csv_file = "gesture_data.csv"
        self.model_file = "gesture_classifier.pkl"
        self.current_label = None
        self.is_recording = False
        self.mode = "train"  # "train" or "recognize"
        self.classifier = None
        self.prediction_buffer = deque(maxlen=5)
        self.recognized_text = ""
        self.last_recognized_text = ""  # Track last gesture to avoid duplicate callbacks
        self.sample_count = 0
        
        # Activation gesture pattern (OK Jarvis -> Command)
        self.is_activated = False
        self.activation_time = None
        self.command_buffer = []
        self.confirmed_command = ""
        self.last_confirmed_command = ""  # Track last command to avoid duplicates
        
        # Track initial sample count from CSV (to preserve trained data)
        self.initial_sample_count = self.get_csv_sample_count()
        
        # Camera
        self.cap = None
        self.running = True
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize camera
        self.init_camera()
        
        # Start camera update loop (using tkinter's after method - no threading)
        self.update_camera()
        
    def init_camera(self):
        """Initialize camera with error handling for macOS"""
        try:
            # Try different camera indices (macOS sometimes uses different indices)
            for index in [0, 1]:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Camera initialized successfully on index {index}")
                    return
                self.cap.release()
            
            # If we get here, no camera was found
            messagebox.showerror("Camera Error", 
                "Cannot access camera!\n\n"
                "On macOS, please check:\n"
                "1. System Preferences > Security & Privacy > Camera\n"
                "2. Grant permission to Terminal/Python\n"
                "3. Restart the application")
            self.running = False
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize camera:\n{str(e)}")
            self.running = False
        
    def setup_gui(self):
        # Main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left side - Camera feed
        left_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.camera_label = ttk.Label(left_frame)
        self.camera_label.grid(row=0, column=0)
        
        # Status display
        self.status_label = ttk.Label(left_frame, text="Status: Ready", 
                                     font=("Arial", 12, "bold"),
                                     foreground="green")
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Right side - Controls
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(right_frame, text="Mode", padding="10")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.mode_var = tk.StringVar(value="train")
        ttk.Radiobutton(mode_frame, text="Training Mode", variable=self.mode_var, 
                       value="train", command=self.switch_mode).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(mode_frame, text="Recognition Mode", variable=self.mode_var, 
                       value="recognize", command=self.switch_mode).grid(row=1, column=0, sticky=tk.W)
        
        # Training controls
        self.train_frame = ttk.LabelFrame(right_frame, text="Training Controls", padding="10")
        self.train_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.train_frame, text="Text Phrase:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.phrase_entry = ttk.Entry(self.train_frame, width=30)
        self.phrase_entry.grid(row=1, column=0, pady=2)
        
        ttk.Button(self.train_frame, text="Set Phrase", 
                  command=self.set_phrase).grid(row=2, column=0, pady=5)
        
        self.current_phrase_label = ttk.Label(self.train_frame, text="Current: None", 
                                             font=("Arial", 10, "bold"))
        self.current_phrase_label.grid(row=3, column=0, pady=5)
        
        self.record_btn = ttk.Button(self.train_frame, text="🔴 Start Recording (R)", 
                                    command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=4, column=0, pady=5)
        
        self.sample_count_label = ttk.Label(self.train_frame, text="Samples: 0")
        self.sample_count_label.grid(row=5, column=0, pady=2)
        
        ttk.Button(self.train_frame, text="Clear Current Samples", 
                  command=self.clear_current_samples).grid(row=6, column=0, pady=5)
        
        ttk.Button(self.train_frame, text="Train Model", 
                  command=self.train_model).grid(row=7, column=0, pady=10)
        
        # Learned gestures list
        ttk.Label(self.train_frame, text="Learned Gestures:", 
                 font=("Arial", 9, "bold")).grid(row=8, column=0, sticky=tk.W, pady=(10, 5))
        
        # Scrollable frame for gestures
        self.gestures_canvas = tk.Canvas(self.train_frame, height=200, bg="white")
        self.gestures_canvas.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.gestures_scrollbar = ttk.Scrollbar(self.train_frame, orient="vertical", 
                                                command=self.gestures_canvas.yview)
        self.gestures_scrollbar.grid(row=9, column=1, sticky=(tk.N, tk.S))
        self.gestures_canvas.configure(yscrollcommand=self.gestures_scrollbar.set)
        
        self.gestures_frame = ttk.Frame(self.gestures_canvas)
        self.gestures_canvas.create_window((0, 0), window=self.gestures_frame, anchor="nw")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.gestures_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.gestures_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Initial population of gesture list
        self.refresh_gesture_list()
        
        # Recognition display
        self.recog_frame = ttk.LabelFrame(right_frame, text="Recognized Text", padding="10")
        self.recog_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        self.recog_frame.grid_remove()  # Hidden initially
        
        self.recognized_label = ttk.Label(self.recog_frame, text="", 
                                         font=("Arial", 16, "bold"),
                                         foreground="blue",
                                         wraplength=250)
        self.recognized_label.grid(row=0, column=0, pady=10)
        
        ttk.Label(self.recog_frame, text="Show your hand and perform gestures",
                 wraplength=250).grid(row=1, column=0)
        
        # Info panel
        info_frame = ttk.LabelFrame(right_frame, text="Information", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=35, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0)
        self.info_text.insert("1.0", "Training Mode:\n1. Enter a phrase\n2. Click 'Set Phrase'\n3. Perform gesture and press 'R' to record\n4. Collect 30-50 samples per gesture\n5. Click 'Train Model'\n\nRecognition Mode:\n- Perform gestures to see recognized text")
        self.info_text.config(state=tk.DISABLED)
        
        # Keyboard bindings
        self.root.bind('<r>', lambda e: self.toggle_recording())
        self.root.bind('<R>', lambda e: self.toggle_recording())
        
    def switch_mode(self):
        self.mode = self.mode_var.get()
        
        if self.mode == "train":
            self.train_frame.grid()
            self.recog_frame.grid_remove()
            self.status_label.config(text="Mode: TRAINING", foreground="orange")
        else:
            # Load model for recognition
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.train_frame.grid_remove()
                self.recog_frame.grid()
                self.status_label.config(text="Mode: RECOGNITION", foreground="green")
            else:
                messagebox.showerror("Error", "No trained model found! Please train a model first.")
                self.mode_var.set("train")
                self.mode = "train"
    
    def set_phrase(self):
        phrase = self.phrase_entry.get().strip()
        if phrase:
            self.current_label = phrase
            self.current_phrase_label.config(text=f"Current: {phrase}")
            self.record_btn.config(state=tk.NORMAL)
            self.phrase_entry.delete(0, tk.END)
            self.refresh_gesture_list()
            messagebox.showinfo("Success", f"Phrase set to: '{phrase}'\nNow press 'R' or the button to record samples.")
        else:
            messagebox.showwarning("Warning", "Please enter a phrase first!")
    
    def toggle_recording(self):
        if self.mode != "train" or not self.current_label:
            return
            
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.config(text="⏹ Stop Recording")
            self.status_label.config(text="🔴 RECORDING", foreground="red")
        else:
            self.record_btn.config(text="🔴 Start Recording (R)")
            self.status_label.config(text="Mode: TRAINING", foreground="orange")
    
    def get_csv_sample_count(self):
        """Get the current number of samples in the CSV file"""
        if not os.path.exists(self.csv_file):
            return 0
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_file)
            return len(df)
        except:
            return 0
    
    def get_learned_gestures(self):
        """Get dict of unique gesture labels and their counts from CSV"""
        if not os.path.exists(self.csv_file):
            return {}
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_file)
            # Return dict of {label: count}
            result = dict(sorted(df['label'].value_counts().items()))
            return result
        except Exception as e:
            print(f"Error loading gestures: {e}")
            return {}
    
    def refresh_gesture_list(self):
        """Update the display of learned gestures"""
        # Clear existing widgets
        for widget in self.gestures_frame.winfo_children():
            widget.destroy()
        
        gestures_dict = self.get_learned_gestures()
        
        if not gestures_dict:
            ttk.Label(self.gestures_frame, text="(No gestures trained yet)", 
                     foreground="gray").pack(pady=20)
        else:
            # Show total count at top
            total_label = ttk.Label(self.gestures_frame, 
                                   text=f"Total: {len(gestures_dict)} gestures",
                                   font=("Arial", 8, "italic"),
                                   foreground="gray")
            total_label.pack(pady=(5, 10))
            
            for gesture, count in gestures_dict.items():
                row_frame = ttk.Frame(self.gestures_frame)
                row_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Gesture label with count
                ttk.Label(row_frame, text=f"• {gesture} ({count})", 
                         font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
                
                # Delete button
                delete_btn = ttk.Button(row_frame, text="✕", width=3,
                                       command=lambda g=gesture: self.delete_gesture(g))
                delete_btn.pack(side=tk.RIGHT, padx=5)
        
        # Update scroll region
        self.gestures_frame.update_idletasks()
        self.gestures_canvas.configure(scrollregion=self.gestures_canvas.bbox("all"))
    
    def delete_gesture(self, label):
        """Delete all samples for a specific gesture"""
        result = messagebox.askyesno(
            "Delete Gesture",
            f"Delete all training data for '{label}'?\n\nThis cannot be undone."
        )
        
        if not result:
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_file)
            
            # Remove rows with this label
            df_filtered = df[df['label'] != label]
            
            if len(df_filtered) == 0:
                # No data left, delete the file
                os.remove(self.csv_file)
            else:
                # Save filtered data
                df_filtered.to_csv(self.csv_file, index=False)
            
            # Delete model file (needs retraining)
            if os.path.exists(self.model_file):
                os.remove(self.model_file)
                if self.mode == "recognize":
                    # Switch back to training mode
                    self.mode_var.set("train")
                    self.switch_mode()
            
            # Update tracking
            self.initial_sample_count = self.get_csv_sample_count()
            
            # Refresh the list
            self.refresh_gesture_list()
            
            messagebox.showinfo("Success", 
                              f"Gesture '{label}' deleted.\n\nModel reset - please retrain.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete gesture:\n{str(e)}")
    
    def clear_current_samples(self):
        """Clear samples recorded in this session (keep previously trained data)"""
        if self.sample_count == 0:
            messagebox.showinfo("Info", "No new samples to clear.")
            return
        
        result = messagebox.askyesno(
            "Clear Samples",
            f"Clear {self.sample_count} samples recorded in this session?\n\n"
            f"This will keep your {self.initial_sample_count} previously saved samples."
        )
        
        if not result:
            return
        
        try:
            if self.initial_sample_count == 0:
                # Just delete the file if there were no initial samples
                if os.path.exists(self.csv_file):
                    os.remove(self.csv_file)
            else:
                # Keep only the first N rows (initial samples)
                import pandas as pd
                df = pd.read_csv(self.csv_file)
                df_keep = df.head(self.initial_sample_count)
                df_keep.to_csv(self.csv_file, index=False)
            
            # Reset counter
            self.sample_count = 0
            self.sample_count_label.config(text="Samples: 0")
            self.refresh_gesture_list()
            messagebox.showinfo("Success", "Current session samples cleared!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear samples:\n{str(e)}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmark coordinates"""
        landmarks = []
        wrist = hand_landmarks.landmark[0]
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        return landmarks
    
    def save_sample(self, landmarks, label):
        """Save a sample to the CSV file"""
        file_exists = os.path.exists(self.csv_file)
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                header = []
                for i in range(21):
                    header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
                header.append('label')
                writer.writerow(header)
            
            row = landmarks + [label]
            writer.writerow(row)
    
    def trigger_callback(self, gesture_text, confidence=1.0):
        """
        Trigger the callback function with the recognized gesture
        
        Args:
            gesture_text: The recognized gesture text
            confidence: Confidence score (0.0 to 1.0)
        """
        if self.on_gesture_callback:
            try:
                self.on_gesture_callback(gesture_text, confidence)
            except Exception as e:
                print(f"Error in gesture callback: {e}")
    
    def update_camera(self):
        """Camera update loop using tkinter's after method (no flickering!)"""
        if not self.running or self.cap is None:
            return
            
        success, frame = self.cap.read()
        if not success:
            # Try again in 10ms
            self.root.after(10, self.update_camera)
            return
        
        # Flip and process
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands (solutions API)
        results = self.hands.process(rgb_frame)
        
        # Draw landmarks and handle logic
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = self.extract_landmarks(hand_landmarks)
                
                # Training mode
                if self.mode == "train" and self.is_recording and self.current_label:
                    self.save_sample(landmarks, self.current_label)
                    self.sample_count += 1
                    self.sample_count_label.config(text=f"Samples: {self.sample_count}")
                    cv2.putText(frame, f"RECORDING: {self.sample_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Recognition mode
                elif self.mode == "recognize" and self.classifier:
                    prediction = self.classifier.predict([landmarks])[0]
                    self.prediction_buffer.append(prediction)
                    
                    if len(self.prediction_buffer) >= 3:
                        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                        self.recognized_text = most_common
                        self.recognized_label.config(text=self.recognized_text)
                        
                        # Trigger callback if gesture changed
                        if most_common != self.last_recognized_text:
                            # Calculate confidence (percentage of buffer that agrees)
                            confidence = self.prediction_buffer.count(most_common) / len(self.prediction_buffer)
                            self.trigger_callback(most_common, confidence)
                            self.last_recognized_text = most_common
                        
                        # Display real-time recognition in GREEN
                        cv2.putText(frame, self.recognized_text, (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        # ACTIVATION PATTERN LOGIC
                        # Step 1: Check if "OK Jarvis" was recognized
                        if most_common == "OK Jarvis" and not self.is_activated:
                            self.is_activated = True
                            self.activation_time = time.time()
                            self.command_buffer = []
                            self.confirmed_command = "Listening..."
                            print("DEBUG: Activated! Listening for commands...")
                        
                        # Step 2: Collect commands during 3-second window
                        elif self.is_activated:
                            elapsed = time.time() - self.activation_time
                            
                            if elapsed < 3.0:
                                # Still collecting - add non-activation gestures to buffer
                                if most_common != "OK Jarvis":
                                    self.command_buffer.append(most_common)
                                # Keep showing "Listening..."
                            else:
                                # Window closed - determine predominant command
                                if len(self.command_buffer) >= 2:  # Need at least 2 samples
                                    predominant = Counter(self.command_buffer).most_common(1)[0][0]
                                    self.confirmed_command = predominant
                                    
                                    # Trigger callback for confirmed command (if changed)
                                    if predominant != self.last_confirmed_command:
                                        command_confidence = self.command_buffer.count(predominant) / len(self.command_buffer)
                                        self.trigger_callback(f"COMMAND: {predominant}", command_confidence)
                                        self.last_confirmed_command = predominant
                                    
                                    print(f"DEBUG: Confirmed command: {predominant}")
                                else:
                                    self.confirmed_command = "(no command detected)"
                                    print("DEBUG: No clear command detected")
                                
                                # Reset activation
                                self.is_activated = False
                                self.command_buffer = []
                        
                        # Display confirmed command in BLUE below green text
                        if self.confirmed_command:
                            cv2.putText(frame, f"CMD: {self.confirmed_command}", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        else:
            if self.mode == "recognize":
                self.prediction_buffer.clear()
                self.recognized_text = ""
                self.recognized_label.config(text="(show hand)")
        
        # Convert for tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update display (safe - we're in main thread)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
        
        # Schedule next update (30 FPS = ~33ms delay)
        self.root.after(33, self.update_camera)
    
    def train_model(self):
        """Train the classifier on collected data"""
        if not os.path.exists(self.csv_file):
            messagebox.showerror("Error", "No training data found! Please collect samples first.")
            return
        
        try:
            import pandas as pd
            
            df = pd.read_csv(self.csv_file)
            
            if len(df) < 10:
                messagebox.showwarning("Warning", f"Only {len(df)} samples found. Collect at least 30+ samples per gesture for best results.")
            
            X = df.drop('label', axis=1).values
            y = df['label'].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            accuracy = clf.score(X_test, y_test)
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(clf, f)
            
            self.refresh_gesture_list()
            
            messagebox.showinfo("Success", 
                              f"Model trained successfully!\n\n"
                              f"Accuracy: {accuracy*100:.1f}%\n"
                              f"Training samples: {len(X_train)}\n"
                              f"Test samples: {len(X_test)}\n\n"
                              f"You can now switch to Recognition Mode!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        cv2.destroyAllWindows()
        self.root.destroy()


# ============================================================================
# EXAMPLE USAGE WITH CALLBACK
# ============================================================================

def my_action_handler(gesture_text, confidence):
    """
    This is your custom function that gets called when a gesture is recognized
    
    Args:
        gesture_text: The text of the recognized gesture (e.g., "Thumbs Up", "Peace Sign")
        confidence: How confident the model is (0.0 to 1.0)
    """
    print(f"\n{'='*50}")
    print(f"GESTURE DETECTED: {gesture_text}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"{'='*50}")
    
    # Example: Take different actions based on the gesture
    if "COMMAND:" in gesture_text:
        # This is a confirmed command after "OK Jarvis"
        command = gesture_text.replace("COMMAND: ", "")
        print(f"→ Executing command: {command}")
        
        # Add your command logic here
        if command == "Volume Up":
            print("  [ACTION] Increasing volume...")
            # os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
        
        elif command == "Volume Down":
            print("  [ACTION] Decreasing volume...")
            # os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
        
        elif command == "Next Channel":
            print("  [ACTION] Switching to next channel...")
            # Your TV control code here
        
        elif command == "Previous Channel":
            print("  [ACTION] Switching to previous channel...")
            # Your TV control code here
    
    else:
        # This is a regular gesture recognition (not a command)
        print(f"→ Regular gesture: {gesture_text}")
        
        # Example: Log gestures, update UI, etc.
        if gesture_text == "Peace Sign":
            print("  [INFO] User made peace sign")
        
        elif gesture_text == "Thumbs Up":
            print("  [INFO] User likes this!")


def main():
    # Check dependencies
    try:
        import pandas as pd
        import sklearn
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nPlease install requirements:")
        print("pip install opencv-python mediapipe scikit-learn pandas pillow")
        return
    
    root = tk.Tk()
    
    # Create the GUI with the callback function
    app = GestureToTextGUI(root, on_gesture_callback=my_action_handler)
    
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()


if __name__ == "__main__":
    main()