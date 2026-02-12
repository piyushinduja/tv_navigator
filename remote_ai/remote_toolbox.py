from tools import ToolBox
from .remote_controls import RemoteControls

remote_toolbox = ToolBox()
remote_controls = RemoteControls()

@remote_toolbox.tool
def play():
    """
    Call this function to play the video
    """
    remote_controls.play()

@remote_toolbox.tool
def pause():
    """
    Call this function to pause the video
    """
    remote_controls.pause()

@remote_toolbox.tool
def mute_unmute():
    """
    Call this function to mute or unmute the video
    """
    remote_controls.mute_unmute()

@remote_toolbox.tool
def volume_up(amount: int = 1):
    """
    Increase the volume
    
    Args:
        amount: Amount to increase the volume by (default: 1)
    """
    remote_controls.volume_up(amount)
        
@remote_toolbox.tool
def volume_down(amount: int = 1):
    """
    Decrease the volume

    Args:
        amount: Amount to decrease the volume by (default: 1)
    """
    remote_controls.volume_down(amount)

@remote_toolbox.tool
def skip_forward(seconds: int = 5):
    """
    Skip forward in the YouTube video. Use this when the user wants to fast forward or skip ahead.
    
    Args:
        seconds: Number of seconds to skip forward (default: 5)
    """
    remote_controls.skip_forward(seconds)

@remote_toolbox.tool
def skip_back(seconds: int = 5):
    """
    Skip backward in the YouTube video. Use this when the user wants to rewind or go back.
    
    Args:
        seconds: Number of seconds to skip backward (default: 5)
    """
    remote_controls.skip_back(seconds)
