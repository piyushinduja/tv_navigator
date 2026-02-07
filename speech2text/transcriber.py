import speech_recognition as sr
import threading
from remote_ai.main import ai_agent
import asyncio

WAKE_WORDS = [f"{word} jarvis" for word in ["hey", "ok", "hello", "hi", "okay"]]
WAKE_WORDS.append("jarvis")

r = sr.Recognizer()
mic = sr.Microphone()

listening_enabled = threading.Event()
command_done = threading.Event()

listening_enabled.set()   # start in passive mode


def passive_listener():
    print("🎧 Passive listening started")

    with mic as source:
        r.adjust_for_ambient_noise(source)

    while True:
        listening_enabled.wait()  # waiting for the wake word to be detected

        try:
            with mic as source:
                audio = r.listen(source, timeout=1, phrase_time_limit=10) # timeout: stop listening if no speech, phrase_time_limit: max length of speech

            text = r.recognize_google(audio).lower()
            print(f"👂 Heard: {text}")

            if any(wake_word in text for wake_word in WAKE_WORDS):
                print("🔥 Wake word detected!")
                print("📤 Sending text to AI agent:", text)
                listening_enabled.clear() # Stop listening immediately
                command_done.clear() # Clear previous signal
                send_to_ai_agent(text) # Send text to external AI agent
                command_done.wait() # Wait for AI agent to signal completion (with timeout)
                print("🔁 AI done. Resuming passive listening\n")
                listening_enabled.set()

        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print("Error:", e)


def send_to_ai_agent(text):
    """
    This function should send text to your friend's AI agent.
    The AI agent MUST call `on_command_done()` when finished.
    """
    print("⚙️ AI agent processing started...")
    asyncio.run(ai_agent(text)) # Call the AI agent with the text
    command_done.set()  # Simulate immediate completion for testing


if __name__ == "__main__":
    passive_listener()
