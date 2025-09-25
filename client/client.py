import requests
import time
import os
import argparse
import sys
from datetime import datetime

# API endpoints
API_URL = "-"
# LED control URLs
LED_BOTH_ON_URL = "http://127.0.0.1:5050/U_ON_B_ON"
LED_BOTH_OFF_URL = "http://127.0.0.1:5050/B_OFF_U_OFF"
LED_U_ON_B_OFF_URL = "http://127.0.0.1:5050/U_ON_B_OFF"
LED_B_ON_U_OFF_URL = "http://127.0.0.1:5050/B_ON_U_OFF"
TRANSCRIBE_URL = "-"

# ANSI color codes for colorized output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

class ColorizedLogger:
    """
    A simple colorized logger for console output
    """
    @staticmethod
    def info(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.BLUE}[INFO]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")
        
    @staticmethod
    def success(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")
        
    @staticmethod
    def warning(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")
        
    @staticmethod
    def error(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")
        
    @staticmethod
    def debug(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.MAGENTA}[DEBUG]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")
        
    @staticmethod
    def highlight(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.CYAN}[HIGHLIGHT]{Colors.RESET} {Colors.BOLD}[{timestamp}]{Colors.RESET} {message}")

# Initialize logger
logger = ColorizedLogger()

def control_led(state):
    """
    Control the LED state.
    
    Args:
        state: String indicating LED state to set
            - "on": Turn both LEDs on
            - "off": Turn both LEDs off
            - "u_on": Turn only U LED on
            - "b_on": Turn only B LED on
    """
    try:
        if state.lower() == "on":
            logger.info("Turning both LEDs on...")
            response = requests.post(LED_BOTH_ON_URL, timeout=5)
            response.raise_for_status()
            logger.success("Both LEDs turned on successfully")
        elif state.lower() == "off":
            logger.info("Turning both LEDs off...")
            response = requests.post(LED_BOTH_OFF_URL, timeout=5)
            response.raise_for_status()
            logger.success("Both LEDs turned off successfully")
        elif state.lower() == "u_on":
            logger.info("Turning only U LED on...")
            response = requests.post(LED_U_ON_B_OFF_URL, timeout=5)
            response.raise_for_status()
            logger.success("U LED turned on successfully (B LED off)")
        elif state.lower() == "b_on":
            logger.info("Turning only B LED on...")
            response = requests.post(LED_B_ON_U_OFF_URL, timeout=5)
            response.raise_for_status()
            logger.success("B LED turned on successfully (U LED off)")
        else:
            logger.warning(f"Unknown LED state: {state}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error controlling LED: {str(e)}")
        
def send_value(v: int):
    """
    Send a value to the server to control LED state.
    
    Args:
        v: 1 to turn on, 0 to turn off
    """
    try:
        logger.info(f"Sending value {v} to server...")
        r = requests.post(API_URL, json={"value": v}, timeout=5)
        r.raise_for_status()
        led = r.json().get("led")
        logger.highlight(f"LED status from server: {led}")
        if led == "on":
            control_led("on")
        elif led == "off":
            control_led("off")
        else:
            logger.warning(f"Server said: {led}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending value: {str(e)}")

def detect_led_status(timeout=2):
    """
    Detect current LED status by sending a simple request to the LED controller
    
    Args:
        timeout: Request timeout in seconds (default: 2)
    
    Returns:
        String: "on", "off", or "unknown"
    """
    try:
        # Try to get the LED status by sending a simple request to the LED controller
        # Using the same endpoints but with a GET method to just check status
        response = requests.get("http://127.0.0.1:5050/status", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Parse the response to determine LED status
        if "status" in data:
            if data["status"] in ["on", "ON"]:
                return "on"
            elif data["status"] in ["off", "OFF"]:
                return "off"
        
        # Fall back to an alternative method if the first one doesn't work
        return "unknown"
    except requests.exceptions.ConnectionError:
        # Silently fail on connection errors - the LED controller might be offline
        return "unknown"
    except requests.exceptions.Timeout:
        # Silently fail on timeouts - the LED controller might be unresponsive
        return "unknown"
    except requests.exceptions.RequestException as e:
        # Only log other types of request errors
        logger.error(f"Error detecting LED status: {str(e)}")
        return "unknown"

def send_audio_for_transcription(audio_file_path, server_url=TRANSCRIBE_URL, control_led_automatically=True):
    """
    Sends an audio file to the server for transcription and controls LED based on content.
    
    Args:
        audio_file_path: Path to the audio file (WAV format recommended)
        server_url: Server URL for transcription
        control_led_automatically: If True, will control LED based on transcription content
        
    Returns:
        Dictionary containing transcription results
    """
    if not os.path.exists(audio_file_path):
        logger.error(f"File {audio_file_path} not found.")
        return None
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': audio_file}
            logger.info(f"Sending file {audio_file_path} to server...")
            response = requests.post(server_url, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
        
        logger.highlight("Transcription results:")
        
        # Get all fields from the response
        transcribed_text = result.get('text', '')
        command = result.get('command', 'unknown')
        explanation = result.get('explanation', '')
        led_state = result.get('command', 'unknown') #result.get('led_state', 'unchanged')
        original_text = result.get('original_text', '')
        
        # Log the transcription results
        if transcribed_text:
            logger.info(f"Text: {transcribed_text}")
            logger.debug(f"Language: {result.get('language', 'Unknown')}")
            logger.debug(f"Confidence: {result.get('language_probability', 0):.2f}")
        
        # Log additional information if available
        if command != 'unknown':
            logger.info(f"Server detected command: {command}")
        
        if explanation:
            logger.debug(f"Server explanation: {explanation}")
            
        if original_text and original_text != transcribed_text:
            logger.debug(f"Original text: {original_text}")
        
        # Control LED based on server response or transcribed text if requested
        if control_led_automatically:
            # First priority: Check if the server has specified a LED state to set
            if led_state == 'on':
                logger.info("Server instructed to turn both LEDs on")
                control_led("on")
            elif led_state == 'off':
                logger.info("Server instructed to turn both LEDs off")
                control_led("off")
            elif led_state == 'u_on':
                logger.info("Server instructed to turn only U LED on")
                control_led("u_on")
            elif led_state == 'b_on':
                logger.info("Server instructed to turn only B LED on")
                control_led("b_on")
            # Second priority: Check if the server detected a command
            elif command == 'on':
                logger.info("Using server-detected command to turn both LEDs on")
                control_led("on")
            elif command == 'off':
                logger.info("Using server-detected command to turn both LEDs off")
                control_led("off")
            elif command == 'u_on':
                logger.info("Using server-detected command to turn only U LED on")
                control_led("u_on")
            elif command == 'b_on':
                logger.info("Using server-detected command to turn only B LED on")
                control_led("b_on")
            # Third priority: Fall back to text analysis if no command was provided by the server
            elif transcribed_text:
                text_lower = transcribed_text.lower()
                print("**")
                print(f"Transcribed text for LED control: {text_lower}")
                print("**")
                # Check for LED control phrases
                if any(phrase in text_lower for phrase in ["both on", "all on"]):
                    logger.info("Detected command to turn both LEDs on in transcription")
                    control_led("on")
                elif any(phrase in text_lower for phrase in ["both off", "all off"]):
                    logger.info("Detected command to turn both LEDs off in transcription")
                    control_led("off")
                elif any(phrase in text_lower for phrase in ["u on", "upper on", "blue on"]):
                    logger.info("Detected command to turn only U LED on in transcription")
                    control_led("u_on")
                elif any(phrase in text_lower for phrase in ["b on", "lower on", "red on"]):
                    logger.info("Detected command to turn only B LED on in transcription")
                    control_led("b_on")
                elif any(phrase in text_lower for phrase in ["on", "switch on", "light on", "led on"]):
                    logger.info("Detected general command to turn LEDs on in transcription")
                    control_led("on")
                elif any(phrase in text_lower for phrase in ["turn off", "switch off", "light off", "led off"]):
                    logger.info("Detected command to turn LEDs off in transcription")
                    control_led("off")
                else:
                    logger.debug("No LED control commands detected in transcription")
            
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending audio file: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    # Check if we should disable colors (for environments that don't support ANSI color codes)
    if "--no-color" in sys.argv:
        sys.argv.remove("--no-color")
        # Remove color codes by setting them to empty strings
        for attr in dir(Colors):
            if not attr.startswith("__"):
                setattr(Colors, attr, "")

    parser = argparse.ArgumentParser(description="Audio transcription and LED control client")
    parser.add_argument("--no-color", action="store_true", help="Disable colorized output")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # LED control subcommand
    led_parser = subparsers.add_parser("led", help="Control LED directly")
    led_parser.add_argument("state", choices=["on", "off", "u_on", "b_on"], 
                           help="LED state (on=both on, off=both off, u_on=only U on, b_on=only B on)")
    
    # Value send subcommand
    value_parser = subparsers.add_parser("value", help="Send value to control LED")
    value_parser.add_argument("value", type=int, choices=[0, 1], help="Value to send (0=off, 1=on)")
    
    # Transcribe subcommand
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    transcribe_parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    transcribe_parser.add_argument("--server", default=TRANSCRIBE_URL, help="Server URL for transcription")
    transcribe_parser.add_argument("--no-led-control", action="store_true", 
                                  help="Disable automatic LED control based on transcription")
    transcribe_parser.add_argument("--save", action="store_true", 
                                  help="Save transcription to a text file")
    
    args = parser.parse_args()
    
    logger.info(f"Starting client with command: {args.command if hasattr(args, 'command') else 'help'}")
    
    if args.command == "led":
        control_led(args.state)
    elif args.command == "value":
        send_value(args.value)
    elif args.command == "transcribe":
        logger.info(f"Processing audio file: {args.audio_file}")
        result = send_audio_for_transcription(
            args.audio_file, 
            args.server,
            not args.no_led_control
        )
        
        # Save transcription if requested
        if args.save and result and result.get("text"):
            output_file = os.path.splitext(args.audio_file)[0] + "_transcription.txt"
            with open(output_file, "w") as f:
                f.write(result["text"])
            logger.success(f"Transcription saved to {output_file}")
    else:
        logger.warning("No command specified")
        parser.print_help()
