import requests
import os
import argparse

def send_audio_for_transcription(server_url, audio_file_path):
    """
    Sends an audio file to the server for transcription.
    
    Args:
        server_url: The URL of the transcription endpoint
        audio_file_path: Path to the audio file (WAV format recommended)
        
    Returns:
        Dictionary containing transcription results
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} not found.")
        return None
    
    try:
        files = {'file': open(audio_file_path, 'rb')}
        print(f"Sending file {audio_file_path} to server...")
        response = requests.post(server_url, files=files, timeout=30)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send audio file for transcription")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe",)
    parser.add_argument("--server", default="-", 
                        help="Server URL for transcription")
    
    args = parser.parse_args()
    transcription = send_audio_for_transcription(args.server, args.audio_file)
    
    # If you want to save the transcription to a file
    if transcription and transcription.get("text"):
        output_file = os.path.splitext(args.audio_file)[0] + "_transcription.txt"
        with open(output_file, "w") as f:
            f.write(transcription["text"])
        print(f"Transcription saved to {output_file}")
