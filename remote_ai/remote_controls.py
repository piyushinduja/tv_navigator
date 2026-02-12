import platform
import subprocess
import time
import math
from pynput.keyboard import Controller, Key
from .system_volume import SystemVolume

class RemoteControls:
    def __init__(self):
        self.keyboard = Controller()
        self.system_volume = SystemVolume()
    
    def _focus_youtube_window(self):
        """Bring YouTube browser window to focus"""
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # AppleScript to find and focus window with "YouTube" in title
            script = '''
            tell application "System Events"
                set frontmost of first process whose name contains "Chrome" or name contains "Safari" or name contains "Firefox" to true
            end tell
            '''
            subprocess.run(['osascript', '-e', script])
            time.sleep(0.2)  # Small delay to ensure window is focused
            
        elif system == 'Windows':
            import pygetwindow as gw
            
            # Find window with "YouTube" in title
            youtube_windows = gw.getWindowsWithTitle('YouTube')
            
            if youtube_windows:
                youtube_windows[0].activate()
                time.sleep(0.2)
            else:
                # Try to find any browser window
                browsers = ['Chrome', 'Firefox', 'Edge', 'Safari']
                for browser in browsers:
                    windows = gw.getWindowsWithTitle(browser)
                    if windows:
                        windows[0].activate()
                        time.sleep(0.2)
                        break
    
    def play(self):
        """
        Call this function to play the video
        """
        print("Called play")
        self._focus_youtube_window()
        self.keyboard.press('k')
        self.keyboard.release('k')
    
    def pause(self):
        """
        Call this function to pause the video
        """
        print("Called pause")
        self.play()
    
    def mute_unmute(self):
        """
        Call this function to mute or unmute the video
        """
        print("Called mute/unmute")
        self.system_volume.toggle_mute()
        # keyboard.press('m')
        # keyboard.release('m')
    
    def volume_up(self, amount: int = 1):
        """
        Increase the volume
        
        Args:
            amount: Amount to increase the volume by (default: 1)
        """
        print(f"Called volume up, {amount}")
        for _ in range(amount):
            self.system_volume.increase()
            # keyboard.press(Key.up)
            # keyboard.release(Key.up)
    
    def volume_down(self, amount: int = 1):
        """
        Decrease the volume

        Args:
            amount: Amount to decrease the volume by (default: 1)
        """
        print(f"Called volume down, {amount}")
        for _ in range(amount):
            self.system_volume.decrease()
            # keyboard.press(Key.down)
            # keyboard.release(Key.down)
    
    def skip_forward(self, seconds: int = 5):
        """
        Skip forward in the YouTube video. Use this when the user wants to fast forward or skip ahead.
        
        Args:
            seconds: Number of seconds to skip forward (default: 5)
        """
        print(f"Called skip forward {seconds} seconds")
        self._focus_youtube_window()
        num_of_skips = math.floor(seconds / 5)
        for _ in range(num_of_skips):
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)
    
    def skip_back(self, seconds: int = 5):
        """
        Skip backward in the YouTube video. Use this when the user wants to rewind or go back.
        
        Args:
            seconds: Number of seconds to skip backward (default: 5)
        """
        print(f"Called skip back {seconds} seconds")
        self._focus_youtube_window()
        num_of_skips = math.floor(seconds / 5)
        for _ in range(num_of_skips):
            self.keyboard.press(Key.left)    
            self.keyboard.release(Key.left)