from tools import ToolBox
from .system_volume import SystemVolume
import platform
import subprocess
import keyboard
import time
import math

from pynput.keyboard import Controller, Key

keyboard = Controller()

remote_toolbox = ToolBox()
system_volume = SystemVolume()

def get_platform():
    return platform.system()  # Returns 'Darwin' for Mac, 'Windows' for Windows

def focus_youtube_window():
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


@remote_toolbox.tool
def play():
    """
    Call this function to play the video
    """
    print("Called play")
    focus_youtube_window()
    keyboard.press('k')
    keyboard.release('k')

@remote_toolbox.tool
def pause():
    """
    Call this function to pause the video
    """
    print("Called pause")
    play()

@remote_toolbox.tool
def mute_unmute():
    """
    Call this function to mute or unmute the video
    """
    print("Called mute/unmute")
    system_volume.toggle_mute()

@remote_toolbox.tool
def volume_up(amount: int = 1):
    """
    Increase the volume
    
    Args:
        amount: Amount to increase the volume by (default: 1)
    """
    print(f"Called volume up, {amount}")
    for _ in range(amount):
        system_volume.increase()

@remote_toolbox.tool
def volume_down(amount: int = 1):
    """
    Decrease the volume

    Args:
        amount: Amount to decrease the volume by (default: 1)
    """
    print(f"Called volume down, {amount}")
    for _ in range(amount):
        system_volume.decrease()

@remote_toolbox.tool
def skip_forward(seconds: int = 5):
    """
    Skip forward in the YouTube video. Use this when the user wants to fast forward or skip ahead.
    
    Args:
        seconds: Number of seconds to skip forward (default: 5)
    """
    print(f"Called skip forward {seconds} seconds")
    focus_youtube_window()
    num_of_skips = math.floor(seconds / 5)
    for _ in range(num_of_skips):
        keyboard.press(Key.right)
        keyboard.release(Key.right)

@remote_toolbox.tool
def skip_back(seconds: int = 5):
    """
    Skip backward in the YouTube video. Use this when the user wants to rewind or go back.
    
    Args:
        seconds: Number of seconds to skip backward (default: 5)
    """
    print(f"Called skip back {seconds} seconds")
    focus_youtube_window()
    num_of_skips = math.floor(seconds / 5)
    for _ in range(num_of_skips):
        keyboard.press(Key.left)    
        keyboard.release(Key.left)
