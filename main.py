"""
Production-ready main application for gesture-controlled TV navigator

Features:
- Proper async/sync bridge between tkinter and asyncio
- Error handling and logging
- Graceful shutdown
- Thread-safe operation

Usage:
    python main.py
"""

import tkinter as tk
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue
import sys
import os

# Add parent directory to path if running from subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture.gesture_gui import GestureToTextGUI
from remote_ai.remote_agent import RemoteAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncBridge:
    """Bridge between tkinter (sync) and asyncio (async)"""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def start(self):
        """Start the async event loop in a background thread"""
        def run_async_loop(loop):
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                loop.close()
        
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=run_async_loop, args=(self.loop,), daemon=True)
        self.thread.start()
        logger.info("Async event loop started")
    
    def run_coroutine(self, coro):
        """Run a coroutine in the async event loop"""
        if self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future
        else:
            logger.error("Async loop is not running")
            return None
    
    def stop(self):
        """Stop the async event loop"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            logger.info("Async event loop stopped")


class GestureRemoteApp:
    """Main application integrating gesture recognition with remote control"""
    
    def __init__(self):
        logger.info("Initializing Gesture Remote App")
        
        # Create tkinter root
        self.root = tk.Tk()
        self.root.title("TV Navigator - Gesture Control")
        
        # Initialize async bridge
        self.bridge = AsyncBridge()
        self.bridge.start()
        
        # Initialize remote agent
        self.agent = RemoteAgent()
        
        # Gesture processing queue (for rate limiting)
        self.gesture_queue = Queue(maxsize=10)
        self.last_gesture = None
        self.processing = False
        
        # Create gesture GUI with callback
        self.app = GestureToTextGUI(
            self.root, 
            on_gesture_callback=self.on_gesture_detected
        )
        
        logger.info("Application initialized successfully")
    
    def on_gesture_detected(self, gesture_text, confidence):
        """
        Callback when a gesture is detected
        Runs in the main tkinter thread
        """
        # Log the detection
        logger.info(f"Gesture detected: '{gesture_text}' (confidence: {confidence*100:.1f}%)")
        
        # Filter low confidence gestures
        if confidence < 0.6:
            logger.debug(f"Ignoring low confidence gesture: {confidence*100:.1f}%")
            return
        
        # Avoid processing the same gesture repeatedly
        if gesture_text == self.last_gesture:
            return
        
        self.last_gesture = gesture_text
        
        # Queue the gesture for processing
        try:
            self.gesture_queue.put_nowait((gesture_text, confidence))
            self.process_gesture_queue()
        except:
            logger.warning("Gesture queue full, dropping gesture")
    
    def process_gesture_queue(self):
        """Process gestures from the queue"""
        if self.processing or self.gesture_queue.empty():
            return
        
        self.processing = True
        gesture_text, confidence = self.gesture_queue.get()
        
        # Schedule async processing
        future = self.bridge.run_coroutine(
            self.evaluate_gesture(gesture_text, confidence)
        )
        
        if future:
            # Set callback when done
            future.add_done_callback(lambda f: self.on_evaluation_complete(f))
    
    async def evaluate_gesture(self, gesture_text, confidence):
        """
        Async evaluation of gesture
        Runs in the async event loop thread
        """
        try:
            logger.info(f"Evaluating gesture: {gesture_text}")
            result = await self.agent.evaluate(gesture_text)
            logger.info(f"Evaluation complete: {result}")
            return result
        except Exception as e:
            logger.error(f"Error evaluating gesture '{gesture_text}': {e}", exc_info=True)
            return None
    
    def on_evaluation_complete(self, future):
        """
        Callback when async evaluation completes
        Runs in the async thread, so we need to be careful
        """
        self.processing = False
        
        try:
            result = future.result()
            logger.debug(f"Evaluation result: {result}")
        except Exception as e:
            logger.error(f"Error in evaluation future: {e}")
        
        # Process next gesture if available
        self.root.after(100, self.process_gesture_queue)
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Shutting down application...")
        
        # Cleanup gesture app
        if self.app:
            try:
                self.app.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up gesture app: {e}")
        
        # Stop async bridge
        if self.bridge:
            try:
                self.bridge.stop()
            except Exception as e:
                logger.error(f"Error stopping async bridge: {e}")
        
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
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            print("\nPlease install requirements:")
            print("pip install opencv-python mediapipe scikit-learn pandas pillow")
            return 1
        
        # Create and run app
        app = GestureRemoteApp()
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