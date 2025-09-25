from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
import cv2
import numpy as np
import os
import logging
from datetime import datetime
import requests
import uvicorn
from faster_whisper import WhisperModel
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import re
import yaml
import base64
import io
import subprocess
import sys

# Check for and install required packages
def ensure_package_installed(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        logging.info(f"Package '{import_name}' is available")
    except ImportError:
        logging.warning(f"Package '{import_name}' not found. Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logging.info(f"Package '{package_name}' installed")
        except Exception as e:
            logging.error(f"Failed to install '{package_name}': {e}")
            raise

# Ensure PIL is available (needed for image processing)
ensure_package_installed("pillow", "PIL")

# Now we can safely import from PIL
from PIL import Image

# Import prompts from the separate file
from prompt_templates import LAMP_COMMAND_PROMPTS

# Ensure yaml package is available
ensure_package_installed("pyyaml", "yaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'visionpi.log'))  # Log to file
    ]
)
logger = logging.getLogger("VisionPi_IOT")

# Global variables to store the models
whisper_model = None
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_processor = None

# Load YAML config for VLM prompts
def load_vlm_config():
    config_path = os.path.join(os.path.dirname(__file__), 'vlm_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"VLM config file not found at {config_path}")
        # Return default config if file doesn't exist
        return {
            "prompts": {
                "lamp_detection": "Look at this image. Is there a lamp in the scene? If yes, is it turned on or off? Respond with only 'on', 'off', or 'no lamp'."
            }
        }

vlm_config = load_vlm_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global whisper_model, llm_model, llm_tokenizer, vlm_model, vlm_processor
    
    # Load Whisper model
    logger.info("Loading Whisper model...")
    model_size = "large-v3"
    try:
        whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
        logger.info("Whisper model loaded on GPU")
    except Exception as e:
        logger.error(f"Failed to load model on GPU: {e}")
        logger.info("Falling back to CPU")
        whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded on CPU")
    
    # Load Qwen LLM model
    logger.info("Loading Qwen LLM model...")
    try:
        # Try to import accelerate first
        try:
            import accelerate
            logger.info("Accelerate package is available")
        except ImportError:
            logger.warning("Accelerate package not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "accelerate"])
            logger.info("Accelerate package installed")
        
        model_name = "Qwen/Qwen3-8B"
        logger.info(f"Loading tokenizer for {model_name}")
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Try loading on GPU first, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Attempting to load LLM model on {device}")
        
        if device == "cuda":
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",  # This requires accelerate
                dtype=torch.float16,  # Changed from torch_dtype to dtype
                trust_remote_code=True
            )
            logger.info("Qwen LLM model loaded on GPU")
        else:
            # For CPU loading, avoid using device_map
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            logger.info("Qwen LLM model loaded on CPU")
    except Exception as e:
        logger.error(f"Failed to load Qwen LLM model: {e}")
        llm_model = None
        llm_tokenizer = None
    
    # Load Qwen VLM model
    logger.info("Loading Qwen Vision-Language model...")
    try:
        vlm_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"Loading VLM processor for {vlm_model_name}")
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
        
        # Try loading on GPU first, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Attempting to load VLM model on {device}")
        
        if device == "cuda":
            vlm_model = AutoModelForCausalLM.from_pretrained(
                vlm_model_name,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("Qwen VLM model loaded on GPU")
        else:
            # For CPU loading, avoid using device_map
            vlm_model = AutoModelForCausalLM.from_pretrained(
                vlm_model_name,
                trust_remote_code=True
            )
            logger.info("Qwen VLM model loaded on CPU")
    except Exception as e:
        logger.error(f"Failed to load Qwen VLM model: {e}")
        vlm_model = None
        vlm_processor = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down models...")
    whisper_model = None
    llm_model = None
    llm_tokenizer = None
    vlm_model = None
    vlm_processor = None


app = FastAPI(lifespan=lifespan)

class NumberRequest(BaseModel):
    value: float

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    command: str = "unknown"  # Added field for detected command
    led_state: str = "unchanged"  # Added field for LED state
    analysis: str = ""  # Added field for LLM analysis

class TextCommandRequest(BaseModel):
    text: str
    language: str = "en"  # Default language is English, can also be "fa" for Persian

class CommandResponse(BaseModel):
    command: str  # "on", "off", or "unknown"
    original_text: str
    explanation: str
    led_state: str  # Added field to report the LED state

class ImageCommandRequest(BaseModel):
    image_base64: str = None

class ImageCommandResponse(BaseModel):
    command: str  # "on", "off", "no lamp", or "unknown"
    explanation: str
    led_state: str  # Current LED state



@app.get("/health")
async def health_check():
    """API health check endpoint that returns service status."""
    return {"status": "healthy", "service": "VisionPi_IOT"}



FRAMES_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frames")

@app.websocket("/ws_frames")
async def websocket_frames(websocket: WebSocket):
    await websocket.accept()

    os.makedirs(FRAMES_ROOT, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(FRAMES_ROOT, current_date)
    os.makedirs(date_dir, exist_ok=True)

    frame_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_text("ERR: decode failed")
                continue

            timestamp = datetime.now().strftime("%H-%M-%S-%f")
            frame_filename = f"frame_{timestamp}_{frame_count:04d}.jpg"
            frame_path = os.path.join(date_dir, frame_filename)

            try:
                cv2.imwrite(frame_path, img)
                msg = f"OK {img.shape[1]}x{img.shape[0]} (saved {frame_filename})"
            except Exception as e:
                msg = f"ERR: could not save frame ({e})"

            frame_count += 1
            await websocket.send_text(msg)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = None):
    """
    Transcribe an audio file using Whisper model and analyze with LLM
    
    Args:
        file: The audio file to transcribe
        language: Optional language override (en or fa)
        
    Returns:
        Dictionary with transcription results and command analysis
    """
    if whisper_model is None:
        return {"error": "Model not loaded"}
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
    
    try:
        # Determine language to use (limit to English or Persian)
        task_language = None
        if language:
            if language.lower() in ['en', 'english']:
                task_language = 'en'
                logger.info(f"Using specified language: English")
            elif language.lower() in ['fa', 'persian', 'farsi']:
                task_language = 'fa'
                logger.info(f"Using specified language: Persian")
            else:
                logger.warning(f"Unsupported language specified: {language}. Will detect automatically (limited to en/fa).")
        
        # Transcribe the audio with language constraints
        logger.info("Starting transcription with Whisper model")
        segments, info = whisper_model.transcribe(
            temp_audio_path, 
            beam_size=5,
            language=task_language,  # Use specified language if available
            task="transcribe"  # Explicitly set task to transcription
        )
        
        # Combine all segments into one text
        full_text = " ".join([segment.text for segment in segments])
        
        # If detected language is not English or Persian, handle it
        detected_lang = info.language
        if detected_lang not in ['en', 'fa']:
            logger.warning(f"Detected unsupported language: {detected_lang}. Treating as English.")
            detected_lang = 'en'
            
        logger.info(f"Transcription result: '{full_text}' in language '{detected_lang}'")
        
        # Process the transcribed text with LLM to detect commands
        command_result = process_command_with_llm(full_text, detected_lang)
        
        # Control LED based on the detected command
        led_state = control_led(command_result["command"])
        
        # Create enhanced response
        return {
            "text": full_text,
            "language": detected_lang,
            "language_probability": float(info.language_probability),
            "command": command_result["command"],
            "led_state": led_state,
            "analysis": command_result["explanation"]
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {
            "text": "", 
            "language": "", 
            "language_probability": 0.0,
            "command": "unknown",
            "led_state": "unchanged",
            "analysis": f"Error: {str(e)}"
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

def process_command_with_llm(text, language="en"):
    """
    Process text using the Qwen LLM to determine if it contains a command to turn a lamp on or off.
    
    Args:
        text: The input text to process
        language: The language of the input text (en for English, fa for Persian)
        
    Returns:
        Dictionary with the detected command and explanation
    """
    if llm_model is None or llm_tokenizer is None:
        logger.warning("LLM model not loaded for command processing")
        return {"command": "unknown", 
                "original_text": text,
                "explanation": "LLM model not loaded"}
    
    # Get prompt from templates file, defaulting to English if language not found
    if language.lower() not in LAMP_COMMAND_PROMPTS:
        logger.warning(f"No prompt template found for language {language}, using English")
        language = "en"
    
    # Format the prompt with the input text
    prompt_template = LAMP_COMMAND_PROMPTS[language.lower()]
    prompt = prompt_template.format(text=text)
    
    logger.debug(f"Using prompt for language {language}")

    # Generate response from the LLM
    try:
        logger.info(f"Processing text: '{text}' with LLM")
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        
        # Handle device placement more carefully
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            logger.debug("Generating response from LLM")
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.95,
            )
        
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw LLM response: {response}")
        
        # Extract the decision part (after the prompt)
        decision_text = response[len(prompt):].strip().lower()
        logger.debug(f"Extracted decision text: '{decision_text}'")
        
        # Extract first word which should be "on", "off", or "unknown"
        command = re.search(r'(on|off|unknown)', decision_text)
        if command:
            command = command.group(1)
            logger.info(f"Command detected: '{command}'")
        else:
            command = "unknown"
            logger.warning(f"No valid command found in response, defaulting to 'unknown'")
        
        explanation = f"LLM analyzed the text and determined the command is: {command}"
        
        return {
            "command": command,
            "original_text": text,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error processing with LLM: {str(e)}", exc_info=True)
        return {
            "command": "unknown",
            "original_text": text,
            "explanation": f"Error processing with LLM: {str(e)}"
        }

def control_led(command):
    """
    Control the LED based on the detected command
    
    Args:
        command: The detected command ('on', 'off', or 'unknown')
        
    Returns:
        String indicating the LED state
    """
    logger.info(f"Setting LED state based on command: {command}")
    
    if command == "on":
        # Logic to turn on LED would go here
        # For now, we just return the state
        led_state = "on"
    elif command == "off":
        # Logic to turn off LED would go here
        led_state = "off"
    else:
        # If command is unknown, don't change LED state
        led_state = "unchanged"
    
    return led_state

@app.post("/command", response_model=CommandResponse)
async def process_text_command(request: TextCommandRequest):
    """
    Process a text command to determine if it contains instructions to control a lamp
    and control the LED accordingly
    
    Args:
        request: Text command and language information
        
    Returns:
        Dictionary with the detected command, explanation, and resulting LED state
    """
    logger.info(f"Received command request: '{request.text}' in language '{request.language}'")
    result = process_command_with_llm(request.text, request.language)
    
    # Control LED based on the detected command
    led_state = control_led(result["command"])
    result["led_state"] = led_state
    
    return result

@app.post("/voice-command", response_model=CommandResponse)
async def process_voice_command(file: UploadFile = File(...), language: str = None):
    """
    Process an audio file, transcribe it, and detect if it contains a command to control a lamp,
    and control the LED accordingly
    
    Args:
        file: Audio file to transcribe
        language: Optional language override (en or fa)
        
    Returns:
        Dictionary with the detected command, explanation, and resulting LED state
    """
    logger.info(f"Received voice command with language hint: '{language}'")
    
    if whisper_model is None:
        logger.error("Whisper model not loaded")
        return {
            "command": "unknown", 
            "original_text": "", 
            "explanation": "Whisper model not loaded",
            "led_state": "unchanged"
        }
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
        logger.debug(f"Saved uploaded audio to temporary file: {temp_audio_path}")
    
    try:
        # Normalize language if provided
        task_language = normalize_language(language)
        if task_language:
            logger.info(f"Using specified language: {task_language}")
        
        # Transcribe the audio with language constraints
        logger.info("Transcribing audio with Whisper model")
        segments, info = whisper_model.transcribe(
            temp_audio_path, 
            beam_size=5,
            language=task_language,
            task="transcribe"
        )
        
        # Combine all segments into one text
        transcribed_text = " ".join([segment.text for segment in segments])
        
        # If detected language is not English or Persian, handle it
        detected_lang = info.language
        if detected_lang not in ['en', 'fa']:
            logger.warning(f"Detected unsupported language: {detected_lang}. Treating as English.")
            detected_lang = 'en'
            
        logger.info(f"Transcription result: '{transcribed_text}' in language '{detected_lang}'")
        
        # Process the transcribed text with the LLM
        logger.info(f"Processing transcribed text with LLM")
        result = process_command_with_llm(transcribed_text, detected_lang)
        result["original_text"] = transcribed_text
        
        # Control LED based on the detected command
        led_state = control_led(result["command"])
        result["led_state"] = led_state
        
        return result
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return {
            "command": "unknown", 
            "original_text": "", 
            "explanation": f"Processing failed: {str(e)}",
            "led_state": "unchanged"
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            logger.debug(f"Removing temporary audio file: {temp_audio_path}")
            os.unlink(temp_audio_path)

# Create a simplified endpoint that directly controls LED from voice
@app.post("/voice-to-led")
async def voice_to_led(file: UploadFile = File(...), language: str = None):
    """
    Simplified endpoint that processes audio and returns only LED state
    
    Args:
        file: Audio file to transcribe
        language: Optional language override (en or fa)
        
    Returns:
        Dictionary with LED state
    """
    logger.info(f"Received voice-to-led request with language hint: '{language}'")
    
    # Use the existing voice command processing
    result = await process_voice_command(file, language)
    
    # Return a simplified response with just the LED state
    return {
        "led": result["led_state"],
        "transcript": result["original_text"],
        "command_detected": result["command"]
    }
    # Use the existing voice command processing
    result = await process_voice_command(file, language)
    
    # Return a simplified response with just the LED state
    return {
        "led": result["led_state"],
        "transcript": result["original_text"],
        "command_detected": result["command"]
    }

def normalize_language(language):
    """Helper function to normalize language codes"""
    if not language:
        return None
    
    language = language.lower()
    if language in ['en', 'english']:
        return 'en'
    elif language in ['fa', 'persian', 'farsi']:
        return 'fa'
    else:
        return None

def process_image_with_vlm(image_data):
    """
    Process an image using the Vision-Language Model to detect lamp state
    
    Args:
        image_data: PIL Image or image bytes
        
    Returns:
        Dictionary with the detected command and explanation
    """
    if vlm_model is None or vlm_processor is None:
        logger.warning("VLM model not loaded for image processing")
        return {
            "command": "unknown",
            "explanation": "VLM model not loaded"
        }
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image_data = Image.open(io.BytesIO(image_data))
        
        # Get prompt from config
        prompt = vlm_config["prompts"]["image_captioning"]
        logger.info(f"Processing image with prompt: {prompt}")
        
        # Process image and text inputs
        inputs = vlm_processor(
            text=prompt,
            images=image_data,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = vlm_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        # Decode the response
        response = vlm_processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Raw VLM response: {response}")
        
        # Extract the answer part (after the prompt)
        answer = response.split(prompt)[-1].strip().lower()
        logger.info(f"Extracted answer: {answer}")
        
        # Parse the response to get command
        if "on" in answer:
            command = "on"
        elif "off" in answer:
            command = "off"
        elif "no lamp" in answer:
            command = "no lamp"
        else:
            command = "unknown"
        
        explanation = f"VLM analyzed the image and determined: {answer}"
        
        return {
            "command": command,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error processing image with VLM: {str(e)}", exc_info=True)
        return {
            "command": "unknown",
            "explanation": f"Error processing with VLM: {str(e)}"
        }

@app.post("/image-command", response_model=ImageCommandResponse)
async def process_image_command(file: UploadFile = File(None), request: ImageCommandRequest = None):
    """
    Process an image to detect if it contains a lamp and its state (on/off)
    
    Args:
        file: Image file upload (optional)
        request: JSON request containing base64 encoded image (optional)
        
    Returns:
        Dictionary with detected lamp state and LED control result
    """
    logger.info("Received image command request")
    
    try:
        # Handle image from either file upload or base64 encoded string
        if file is not None:
            # Process uploaded file
            logger.info(f"Processing uploaded image file: {file.filename}")
            image_data = await file.read()
        elif request and request.image_base64:
            # Process base64 encoded image
            logger.info("Processing base64 encoded image")
            image_data = base64.b64decode(request.image_base64)
        else:
            return {
                "command": "unknown",
                "explanation": "No image provided",
            }
        
        # Process image with VLM
        result = process_image_with_vlm(image_data)
        
   
        return {
            "command": result["command"],
            "explanation": result["explanation"],
        }
    
    except Exception as e:
        logger.error(f"Error processing image command: {str(e)}", exc_info=True)
        return {
            "command": "unknown",
            "explanation": f"Error processing image: {str(e)}",
            "led_state": "unchanged"
        }
