# client.py
import requests
import time
import os
API_URL = "-"
LED_ON_URL = "http://127.0.0.1:5050/U_ON_B_ON"
LED_OFF_URL = "http://127.0.0.1:5050/B_OFF_U_OFF"
TRANSCRIBE_URL = "-"

def send_value(v: int):
    r = requests.post(API_URL, json={"value": v}, timeout=5)
    r.raise_for_status()
    led = r.json().get("led")
    print("LED status from server:", led)
    if led == "on":
        requests.post(LED_ON_URL, timeout=5)
    elif led == "off":
        requests.post(LED_OFF_URL, timeout=5)
    else:
        print("Server said:", led)

def send_audio_for_transcription(audio_file_path):
    """
    Sends an audio file to the server for transcription.
    
    Args:
        audio_file_path: Path to the audio file (WAV format recommended)
        
    Returns:
        Dictionary containing transcription results
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} not found.")
        return None
    
    try:
        files = {'file': open(audio_file_path, 'rb')}
        response = requests.post(TRANSCRIBE_URL, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print("Transcription results:")
        print(f"Text: {result.get('text', 'No text returned')}")
        print(f"Language: {result.get('language', 'Unknown')}")
        print(f"Confidence: {result.get('language_probability', 0):.2f}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error sending audio file: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # LED control example
    send_value(1)  # turns on
    time.sleep(2)
    send_value(0)  # turns off
    
    # Audio transcription example
    # Uncomment and modify path to use
    # audio_file = "/path/to/your/audio/file.wav"
    # transcription_result = send_audio_for_transcription(audio_file)
