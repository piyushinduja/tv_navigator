
"""
Main application entry point for gesture-controlled TV navigator

This script integrates:
- Gesture recognition GUI (tkinter)
- Remote controls (synchronous)
- Simple string-to-function mapping

Usage:
    python main.py
"""

import tkinter as tk
import logging
import sys
import os
import speech2text.transcriber as transcriber

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture.gesture_gui import GestureToTextGUI
from remote_ai.remote_controls import RemoteControls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureRemoteApp:
    """Main application integrating gesture recognition with remote controls"""
    
    def __init__(self):
        logger.info("Initializing Gesture Remote App")
        
        # Create tkinter root
        self.root = tk.Tk()
        self.root.title("TV Navigator - Gesture Control")
        
        # Initialize remote controls
        self.controls = RemoteControls()
        
        # Track last gesture to avoid duplicates
        self.last_gesture = None
        
        # Gesture to function mapping
        self.gesture_map = {
            # Playback controls
            "play": self.controls.play,
            "pause": self.controls.pause,
            "play/pause": self.controls.play,
            "stop": self.controls.pause,
            
            # Volume controls
            "volume up": lambda: self.controls.volume_up(1),
            "volume down": lambda: self.controls.volume_down(1),
            "louder": lambda: self.controls.volume_up(2),
            "quieter": lambda: self.controls.volume_down(2),
            "mute": self.controls.mute_unmute,
            "unmute": self.controls.mute_unmute,
            
            # Navigation controls
            "skip forward": lambda: self.controls.skip_forward(5),
            "skip back": lambda: self.controls.skip_back(5),
            "fast forward": lambda: self.controls.skip_forward(10),
            "rewind": lambda: self.controls.skip_back(10),
            "next": lambda: self.controls.skip_forward(30),
            "previous": lambda: self.controls.skip_back(30),
        }
        
        # Create gesture GUI with callback
        self.app = GestureToTextGUI(
            self.root, 
            on_gesture_callback=self.on_gesture_detected
        )
        
        logger.info("Application initialized successfully")
    
    def on_gesture_detected(self, gesture_text, confidence):
        """
        Callback when a gesture is detected
        
        Args:
            gesture_text: The recognized gesture text
            confidence: Confidence score (0.0 to 1.0)
        """
        # Log the detection
        logger.info(f"Gesture: '{gesture_text}' (confidence: {confidence*100:.1f}%)")
        
        # Filter low confidence gestures
        if confidence < 0.6:
            logger.debug(f"Ignoring low confidence gesture: {confidence*100:.1f}%")
            return
        
        # Avoid processing the same gesture repeatedly
        if gesture_text == self.last_gesture:
            return
        
        self.last_gesture = gesture_text
        
        # Handle command pattern (after "OK Jarvis")
        if gesture_text.startswith("COMMAND:"):
            command = gesture_text.replace("COMMAND: ", "").strip()
            self.execute_gesture(command)
        else:
            # Regular gesture
            self.execute_gesture(gesture_text)
    
    def execute_gesture(self, gesture_text):
        """
        Execute the function associated with the gesture
        
        Args:
            gesture_text: The gesture text to execute
        """
        # Normalize the gesture text (lowercase, strip whitespace)
        normalized = gesture_text.lower().strip()
        
        # Try exact match first
        if normalized in self.gesture_map:
            try:
                logger.info(f"Executing: {normalized}")
                self.gesture_map[normalized]()
            except Exception as e:
                logger.error(f"Error executing gesture '{normalized}': {e}", exc_info=True)
        else:
            # Try partial matches
            for key in self.gesture_map:
                if key in normalized or normalized in key:
                    try:
                        logger.info(f"Executing (partial match): {key} for '{normalized}'")
                        self.gesture_map[key]()
                        return
                    except Exception as e:
                        logger.error(f"Error executing gesture '{key}': {e}", exc_info=True)
                        return
            
            # No match found
            logger.warning(f"Unknown gesture: '{gesture_text}'")
            logger.info(f"Available gestures: {', '.join(self.gesture_map.keys())}")
    
    def add_gesture(self, gesture_name, function):
        """
        Add a custom gesture mapping
        
        Args:
            gesture_name: Name of the gesture (case-insensitive)
            function: Function to call when gesture is detected
        """
        self.gesture_map[gesture_name.lower()] = function
        logger.info(f"Added gesture mapping: '{gesture_name}' -> {function.__name__}")
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Shutting down application...")
        
        # Cleanup gesture app
        if self.app:
            try:
                self.app.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up gesture app: {e}")
        
        # Destroy root window
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        
        logger.info("Application shutdown complete")
    
    def run(self):
        """Start the application"""
        logger.info("Starting application...")
        
        # Set cleanup handler
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        
        # Run the tkinter main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.cleanup()


def main():
    """Main entry point"""
    try:
        # Check dependencies
        try:
            import pandas as pd
            import sklearn
            import cv2
            import mediapipe as mp
            from pynput.keyboard import Controller
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            print("\nPlease install requirements:")
            print("pip install opencv-python mediapipe scikit-learn pandas pillow pynput")
            return 1
        
        # Create and run app
        app = GestureRemoteApp()
        
        # Optional: Add custom gesture mappings
        # app.add_gesture("thumbs up", lambda: print("Nice!"))
        # app.add_gesture("peace sign", lambda: print("Peace out!"))
        
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - shutting down")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())