# Image Captioning API

This project provides a FastAPI-based API endpoint that generates descriptive captions for images located within a specified folder. It utilizes a pre-trained Hugging Face Transformers model for the image-to-text task.

## Features

* **Batch Image Captioning**: Process multiple images from a folder.
* **Configurable Model**: Easily change the underlying Hugging Face model.
* **JSON Response**: Returns image paths and their generated captions in a structured JSON format.
* **Error Handling**: Provides feedback on processing errors.
* **Interactive API Docs**: Leverages FastAPI's automatic Swagger UI for easy testing.

## Prerequisites

* Python 3.8+
* `pip` for package management

## Installation

1.  **Clone the repository (if applicable) or download the project files.**
    ```bash
    # If you have a git repository:
    # git clone <your-repository-url>
    # cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required Python libraries:**
    Create a `requirements.txt` file with the following content:

    ```txt
    fastapi[all]
    uvicorn[standard]
    transformers
    Pillow
    torch
    torchvision
    torchaudio
    accelerate
    # Add any other specific dependencies your project might have
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `torch`, `torchvision`, `torchaudio`, and `accelerate` are included as they are common dependencies for Hugging Face Transformers models, especially larger ones. Depending on the exact model and your setup (CPU/GPU), specific versions might be needed.*

## Running the Application

1.  **Ensure your image captioning script is named `main.py` (or adjust the command accordingly).**

2.  **Start the FastAPI application using Uvicorn:**
    ```bash
    uvicorn main:app --reload
    ```
    * `main`: Refers to the Python file `main.py`.
    * `app`: Refers to the FastAPI application instance `app = FastAPI()` within your `main.py`.
    * `--reload`: Enables auto-reloading when code changes (useful for development).

    The application will typically be available at `http://127.0.0.1:8000`.

## API Endpoint

### `POST /caption-images/`

Generates captions for images in a specified folder.

* **Method**: `POST`
* **Request Body**: JSON object specifying the folder location.
    ```json
    {
      "folder_location": "path/to/your/images"
    }
    ```
    Replace `"path/to/your/images"` with the actual relative or absolute path to the folder containing images on the server.

* **Success Response (200 OK)**:
    ```json
    {
      "total_images_found": 1,
      "successfully_captioned": 1,
      "results": [
        {
          "image_path": "/path/to/your/images/example.jpg",
          "description": "a cat sitting on a couch"
        }
      ],
      "message": "Successfully generated captions for all 1 found image(s).",
      "errors": []
    }
    ```

* **Error Responses**:
    * `400 Bad Request`: If the `folder_location` is invalid or not found.
    * `503 Service Unavailable`: If the captioning model failed to load.
    * `500 Internal Server Error`: For other server-side issues.

## How to Test

1.  **Using FastAPI's Interactive Docs:**
    * Once the application is running, navigate to `http://127.0.0.1:8000/docs` in your browser.
    * Find the `POST /caption-images/` endpoint.
    * Click "Try it out".
    * Enter the JSON request body with your image folder path.
    * Click "Execute".

2.  **Using cURL:**
    ```bash
    curl -X POST "[http://127.0.0.1:8000/caption-images/](http://127.0.0.1:8000/caption-images/)" \
         -H "Content-Type: application/json" \
         -d '{"folder_location": "./my_test_images"}'
    ```
    Replace `"./my_test_images"` with the path to your images folder.

## Model Information

This API uses a model from the Hugging Face Transformers library. The default model is specified in `main.py` (e.g., `Salesforce/blip-image-captioning-large`). You can change this to other compatible image-to-text models.

* The first time you run the application, the model will be downloaded, which might take some time and require a stable internet connection.
* Ensure you have enough disk space and memory, especially for larger models.

## Folder Structure (Example)

.├── main.py               # FastAPI application code├── requirements.txt      # Python dependencies├── my_test_images/       # Example folder to store images for captioning│   ├── image1.jpg│   └── image2.png└── venv/                 # Virtual environment (optional, but recommended)
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

## License

This project can be licensed under the MIT License (or choose another license as appropriate).
