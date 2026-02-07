
import platform
import subprocess


class SystemVolume:
    def __init__(self):
        self.platform = platform.system()
        # if self.platform == 'Windows':
        #     from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        #     from comtypes import CLSCTX_ALL
        #     from ctypes import cast, POINTER
            
        #     devices = AudioUtilities.GetSpeakers()
        #     interface = devices.device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        #     self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
    
    def increase(self):
        if self.platform == 'Darwin':
            subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) + 10)'])
        elif self.platform == 'Windows':
            current = self.volume_control.GetMasterVolumeLevelScalar()
            self.volume_control.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
    
    def decrease(self):
        if self.platform == 'Darwin':
            subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) - 10)'])
        elif self.platform == 'Windows':
            current = self.volume_control.GetMasterVolumeLevelScalar()
            self.volume_control.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)

    def toggle_mute(self):
        """Toggle system mute on/off"""
        if self.platform == 'Darwin':
            subprocess.run(['osascript', '-e', 'set volume output muted (not (output muted of (get volume settings)))'])
        elif self.platform == 'Windows':
            current_mute = self.volume_control.GetMute()
            self.volume_control.SetMute(not current_mute, None)