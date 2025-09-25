import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import tempfile
from client import control_led, send_audio_for_transcription, ColorizedLogger, detect_led_status
from pydantic import BaseModel
from typing import Optional

# Initialize logger
logger = ColorizedLogger()

# Make sure the static folder exists
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)
    logger.info(f"Created static folder at {static_folder}")

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Variable to track LED status
led_status = "off"

# Define Pydantic model for image command request
class ImageCommandRequest(BaseModel):
    image_base64: Optional[str] = None

# Define Pydantic model for image command response
class ImageCommandResponse(BaseModel):
    command: str
    explanation: str
    led_state: Optional[str] = "unchanged"

# Import the VLM processing function
def process_image_with_vlm(image_data):
    """
    Process image with Vision Language Model to detect lamp and state
    
    Args:
        image_data: Binary image data
        
    Returns:
        Dictionary with command and explanation
    """
    # This is a placeholder - implement actual VLM processing
    # For now, return a default response
    logger.info("Processing image with VLM (placeholder implementation)")
    
    # In a real implementation, you would:
    # 1. Convert image_data to a format suitable for your VLM
    # 2. Send to the model for processing
    # 3. Interpret the results to determine if a lamp is present and its state
    
    return {
        "command": "unknown", 
        "explanation": "VLM processing not yet implemented"
    }

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

# Ensure serving static files works correctly
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Explicitly serve static files"""
    return send_from_directory('static', filename)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio file for transcription and LED control"""
    global led_status
    
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    # Check if the file actually has content
    file_content = file.read()
    if not file_content or len(file_content) < 100:  # Very small files are likely empty
        logger.error("Empty or invalid audio file received")
        return jsonify({"error": "Empty or invalid audio file"}), 400
    
    # Reset file pointer after reading
    file.seek(0)
    
    # Save the uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    file_path = temp_file.name
    temp_file.close()
    
    try:
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        logger.info(f"Audio file saved temporarily at {file_path}")
        
        # Process the audio file using the existing function
        result = send_audio_for_transcription(file_path, control_led_automatically=True)
        
        if result and "text" in result:
            text = result["text"].lower()
            
            # Update LED status based on the transcription
            if any(phrase in text for phrase in ["turn on", "switch on", "light on", "led on"]):
                led_status = "on"
            elif any(phrase in text for phrase in ["turn off", "switch off", "light off", "led off"]):
                led_status = "off"
            
            return jsonify({
                "text": result["text"],
                "led_status": led_status
            })
        else:
            logger.warning("Transcription failed or returned no text")
            return jsonify({"error": "Failed to transcribe audio", "text": ""}), 200
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Removed temporary file {file_path}")

@app.route('/control_led/<state>', methods=['POST'])
def handle_led_control(state):
    """API endpoint to control LED"""
    global led_status
    
    if state not in ["on", "off"]:
        return jsonify({"success": False, "error": "Invalid state"}), 400
    
    try:
        control_led(state)
        led_status = state
        return jsonify({"success": True, "status": state})
    
    except Exception as e:
        logger.error(f"Error controlling LED: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/led_status', methods=['GET'])
def get_led_status():
    """API endpoint to get current LED status"""
    global led_status
    
    try:
        # Use a very short timeout to prevent hanging
        current_status = detect_led_status(timeout=0.5)
        if current_status != "unknown":
            led_status = current_status
    except Exception as e:
        # Don't log every connection error to reduce log spam
        pass
        
    return jsonify({"status": led_status})

@app.route('/image-command', methods=['POST'])
def process_image_command():
    """
    Process an image to detect if it contains a lamp and its state (on/off)
    """
    global led_status
    logger.info("Received image command request")
    
    try:
        # Debug information
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Files in request: {request.files.keys() if request.files else 'None'}")
        logger.info(f"JSON in request: {bool(request.json)}")
        
        if 'file' not in request.files and not request.json:
            logger.error("No image file or base64 data provided")
            return jsonify({
                "command": "unknown",
                "explanation": "No image provided - expected 'file' in form data or 'image_base64' in JSON"
            }), 400
        
        # Handle image from either file upload or base64 encoded string
        if 'file' in request.files:
            # Process uploaded file
            file = request.files['file']
            if not file.filename:
                logger.error("Empty filename in uploaded file")
                return jsonify({
                    "command": "unknown",
                    "explanation": "Empty filename in uploaded file"
                }), 400
                
            logger.info(f"Processing uploaded image file: {file.filename}")
            image_data = file.read()
            logger.info(f"Read {len(image_data)} bytes from uploaded file")
        elif request.json and 'image_base64' in request.json:
            # Process base64 encoded image
            logger.info("Processing base64 encoded image")
            try:
                image_data = base64.b64decode(request.json['image_base64'])
                logger.info(f"Decoded {len(image_data)} bytes from base64 string")
            except Exception as e:
                logger.error(f"Error decoding base64 image: {str(e)}")
                return jsonify({
                    "command": "unknown",
                    "explanation": f"Error decoding base64 image: {str(e)}"
                }), 400
        else:
            logger.error("No valid image data found in request")
            return jsonify({
                "command": "unknown",
                "explanation": "No valid image provided"
            }), 400
        
        # For debugging, always return a successful result with the image size
        # In production, you would use the real VLM processing function
        dummy_result = {
            "command": "on",  # Hardcoded to 'on' for testing
            "explanation": f"Received image of size {len(image_data)} bytes. This is a dummy response."
        }
        
        # Update LED status based on dummy result (for testing)
        if dummy_result["command"] == "on":
            try:
                control_led("on")
                led_status = "on"
            except Exception as e:
                logger.error(f"Error controlling LED: {str(e)}")
        elif dummy_result["command"] == "off":
            try:
                control_led("off")
                led_status = "off"
            except Exception as e:
                logger.error(f"Error controlling LED: {str(e)}")
        
        return jsonify({
            "command": dummy_result["command"],
            "explanation": dummy_result["explanation"],
            "led_state": led_status
        })
    
    except Exception as e:
        logger.error(f"Error processing image command: {str(e)}", exc_info=True)
        return jsonify({
            "command": "unknown",
            "explanation": f"Error processing image: {str(e)}",
            "led_state": "unchanged"
        }), 500

if __name__ == '__main__':
    logger.info("Starting web server on http://0.0.0.0:5000")
    logger.info(f"Static files will be served from: {static_folder}")
    app.run(host='0.0.0.0', port=5000, debug=True)
