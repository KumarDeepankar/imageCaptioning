# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from transformers import pipeline
from PIL import Image
import logging
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Model used for captioning. You can change this to other compatible models.
# "Salesforce/blip-image-captioning-base" is smaller and faster.
# "Salesforce/blip-image-captioning-large" provides more detail but is slower.
MODEL_NAME = "Salesforce/blip-image-captioning-large"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

# Parameters for the caption generation process
GENERATION_ARGS = {
    "max_length": 150,
    "num_beams": 5,
    "early_stopping": True,
    "repetition_penalty": 1.2,
}

# --- Logging Setup ---
# Configures basic logging for the application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variable for the Hugging Face Model ---
# This will hold the initialized image captioning pipeline.
captioner: Optional[Any] = None

# --- FastAPI App Initialization ---
# Creates a new FastAPI application instance.
app = FastAPI(
    title="Image Captioning API",
    description="An API to generate captions for images in a specified folder.",
    version="1.0.0"
)


# --- Pydantic Models for Request and Response ---
# Defines the structure for the incoming request body.
class ImageCaptionRequest(BaseModel):
    folder_location: str


# Defines the structure for each item in the results list of the response.
class ImageCaptionResponseItem(BaseModel):
    image_path: str
    description: str


# Defines the overall structure of the JSON response.
class CaptionSummaryResponse(BaseModel):
    total_images_found: int
    successfully_captioned: int
    results: List[ImageCaptionResponseItem]
    message: str
    errors: List[str] = []


# --- FastAPI Application Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Initializes the Hugging Face image captioning model.
    This is done once when the application starts to avoid reloading the model on every request.
    """
    global captioner
    logger.info(f"Attempting to initialize Hugging Face pipeline with model: {MODEL_NAME}...")
    logger.info("This might take some time, especially for larger models on the first run...")
    try:
        # Initialize the image-to-text pipeline from Hugging Face Transformers.
        captioner = pipeline("image-to-text", model=MODEL_NAME)
        logger.info(f"Pipeline initialized successfully with model: {MODEL_NAME}.")
        logger.info(f"Pipeline device: {captioner.device}")  # Logs the device (CPU/GPU) the model is running on.
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Hugging Face pipeline during startup: {e}")
        logger.error(
            "The API might not function correctly. Ensure model availability and resources (internet, disk space, "
            "memory).")
        captioner = None  # Ensure captioner is None if initialization fails.


# --- API Endpoint Definition ---
@app.post("/caption-images/", response_model=CaptionSummaryResponse)
async def create_captions_for_images_in_folder(request: ImageCaptionRequest):
    """
    FastAPI endpoint to process images in a given folder and generate captions.

    Args:
        request (ImageCaptionRequest): The request body containing the 'folder_location'.

    Raises:
        HTTPException:
            - 503 Service Unavailable: If the captioning model failed to load.
            - 400 Bad Request: If the provided folder_location is invalid.
            - 500 Internal Server Error: If there's an issue reading the directory.

    Returns:
        CaptionSummaryResponse: A JSON response with captioning results, counts, messages, and errors.
    """
    global captioner
    if captioner is None:
        logger.error("Captioning model is not available. Initialization might have failed during startup.")
        raise HTTPException(status_code=503, detail="Captioning service is unavailable. Model not loaded.")

    folder_path = request.folder_location
    logger.info(f"Received request to caption images in folder: {folder_path}")

    if not os.path.isdir(folder_path):
        logger.error(f"Invalid folder path provided: '{folder_path}' does not exist or is not a directory.")
        raise HTTPException(status_code=400, detail=f"The folder '{folder_path}' does not exist or is not a directory.")

    results: List[ImageCaptionResponseItem] = []
    errors: List[str] = []

    try:
        all_files_in_dir = os.listdir(folder_path)
    except OSError as e:
        logger.error(f"Error listing directory {folder_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read directory contents: {folder_path}")

    # Filter for files with supported image extensions.
    image_filenames = [f for f in all_files_in_dir if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    total_images_found = len(image_filenames)

    if total_images_found == 0:
        message = f"No images with supported extensions {SUPPORTED_EXTENSIONS} found in the folder: {folder_path}"
        logger.info(message)
        return CaptionSummaryResponse(
            total_images_found=0,
            successfully_captioned=0,
            results=[],
            message=message,
            errors=[]
        )

    logger.info(f"Found {total_images_found} image(s) with supported extensions to process in folder: {folder_path}")
    logger.info(f"Using generation parameters for captions: {GENERATION_ARGS}")

    for filename in image_filenames:
        current_image_path_relative = os.path.join(folder_path, filename)
        # Get absolute path for clarity in response, though relative might also work depending on client.
        current_image_path_absolute = os.path.abspath(current_image_path_relative)
        logger.info(f"\n--- Processing image: {filename} ---")

        img: Optional[Image.Image] = None
        try:
            logger.info(f"Attempting to load image: {current_image_path_absolute}...")
            # Open the image file and convert to RGB format.
            img = Image.open(current_image_path_relative).convert("RGB")
            logger.info(f"Image '{filename}' loaded successfully. Mode: {img.mode}, Size: {img.size}")
        except FileNotFoundError:  # Should be rare if os.listdir worked, but defensive.
            msg = f"Image file not found at '{current_image_path_absolute}'. Skipping."
            logger.error(msg)
            errors.append(f"{filename}: {msg}")
            continue
        except Exception as e:
            msg = f"An unexpected error occurred while loading image '{filename}': {e}. Skipping."
            logger.error(msg)
            errors.append(f"{filename}: {msg}")
            continue

        if img is None:  # Should ideally be caught by the try-except block above.
            msg = f"Image object is None after attempting to load '{filename}'. Skipping."
            logger.error(msg)
            errors.append(f"{filename}: {msg}")
            continue

        logger.info(f"Attempting to generate caption for '{filename}'...")
        try:
            # Generate caption using the initialized pipeline and generation arguments.
            captions_output = captioner(img, generate_kwargs=GENERATION_ARGS)

            # The output is typically a list of dictionaries.
            if captions_output and isinstance(captions_output, list) and len(captions_output) > 0:
                first_result = captions_output[0]
                if isinstance(first_result, dict) and 'generated_text' in first_result:
                    generated_text = first_result['generated_text'].strip()
                    logger.info(f"Generated Caption for '{filename}':\n{generated_text}\n")
                    results.append(
                        ImageCaptionResponseItem(image_path=current_image_path_absolute, description=generated_text))
                else:
                    msg = f"Caption output format unexpected for '{filename}'. First element: {first_result}. Skipping."
                    logger.warning(msg)
                    errors.append(f"{filename}: {msg}")
            else:
                msg = f"Captioner returned empty or None output for '{filename}'. Skipping."
                logger.warning(msg)
                errors.append(f"{filename}: {msg}")

        except Exception as e:
            msg = f"An error occurred during caption generation for '{filename}': {e}."
            logger.error(msg, exc_info=True)  # Log the full stack trace for debugging.
            errors.append(f"{filename}: {msg}")
            continue  # Continue to the next image.

    successfully_captioned_count = len(results)

    # Construct an informative summary message based on the processing outcome.
    if successfully_captioned_count == total_images_found and total_images_found > 0:
        message = f"Successfully generated captions for all {total_images_found} found image(s)."
    elif successfully_captioned_count > 0:
        message = f"Generated captions for {successfully_captioned_count} out of {total_images_found} found image(s). See errors for details on failures."
    elif total_images_found > 0 and successfully_captioned_count == 0:  # Found images, but none succeeded
        message = f"Attempted to process {total_images_found} image(s), but no captions were successfully generated. See errors for details."
    else:  # Should be covered by the initial check for total_images_found == 0
        message = "No images were processed or found."

    if not errors and successfully_captioned_count < total_images_found and total_images_found > 0:
        # This case might indicate an issue in error reporting if some images failed silently.
        additional_info = " Some images may have failed processing without explicit error messages being captured in the error list."
        message += additional_info
        logger.warning(
            f"Discrepancy: {total_images_found} images found, {successfully_captioned_count} captioned, but no errors recorded.")

    return CaptionSummaryResponse(
        total_images_found=total_images_found,
        successfully_captioned=successfully_captioned_count,
        results=results,
        message=message,
        errors=errors
    )

