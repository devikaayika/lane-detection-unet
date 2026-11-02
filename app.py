
from pathlib import Path
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Trained U-Net model
MODEL_PATH = BASE_DIR / "models" / "lane_unet_v2.keras"

# Image folders
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

# Model input dimensions
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model = None


def get_model():
    """Load the trained model only when needed."""
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Trained model not found: models/lane_unet_v2.keras"
            )

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )

    return model


def allowed_file(filename):
    """Check the uploaded file extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def detect_lanes(image_path, output_path, threshold=0.5, opacity=0.7):
    """
    Predict lane pixels and blend the predicted lane mask
    with the original road image.
    """

    trained_model = get_model()

    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError(
            "The uploaded image could not be read. "
            "Please select a valid road image."
        )

    height, width = original.shape[:2]

    # Resize to the model's expected input dimensions.
    resized = cv2.resize(
        original,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # OpenCV loads BGR images; training uses RGB.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1].
    input_image = rgb.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Predict lane-pixel probabilities.
    prediction = trained_model.predict(
        input_image,
        verbose=0
    )[0, :, :, 0]

    # Create a binary mask using the selected threshold.
    small_mask = (prediction >= threshold).astype(np.uint8)

    # Resize the mask to the original image dimensions.
    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Blend green only into predicted lane pixels.
    result = original.copy()
    lane_area = mask.astype(bool)

    green = np.zeros_like(original)
    green[:, :] = (0, 255, 0)

    if np.any(lane_area):
        blended = cv2.addWeighted(
            original,
            1.0 - opacity,
            green,
            opacity,
            0
        )
        result[lane_area] = blended[lane_area]

    # Save the result image.
    if not cv2.imwrite(str(output_path), result):
        raise ValueError(
            "Could not save the prediction result."
        )

    return {
        "width": width,
        "height": height,
        "lane_pixels": int(np.count_nonzero(mask)),
        "threshold": threshold,
        "opacity": opacity
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result_url = None
    input_url = None
    summary = None

    # Default slider settings.
    threshold = 0.5
    opacity = 0.7

    if request.method == "POST":
        try:
            # Read interactive slider values.
            threshold = float(
                request.form.get("threshold", "0.5")
            )
            opacity = float(
                request.form.get("opacity", "0.7")
            )

            if not 0.05 <= threshold <= 0.95:
                raise ValueError(
                    "Confidence threshold must be between 0.05 and 0.95."
                )

            if not 0.1 <= opacity <= 1.0:
                raise ValueError(
                    "Overlay strength must be between 0.1 and 1.0."
                )

            # Validate the uploaded file.
            if "image" not in request.files:
                raise ValueError(
                    "Please select a road image."
                )

            file = request.files["image"]

            if not file.filename:
                raise ValueError(
                    "Please select a road image."
                )

            if not allowed_file(file.filename):
                raise ValueError(
                    "Upload a JPG, JPEG, or PNG image."
                )

            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    "The trained model was not found. "
                    "Check models/lane_unet_v2.keras."
                )

            # Generate unique, safe filenames.
            original_name = secure_filename(file.filename)

            if not original_name:
                raise ValueError(
                    "The uploaded filename is invalid."
                )

            unique_name = (
                f"{uuid.uuid4().hex}_{original_name}"
            )

            input_path = UPLOAD_DIR / unique_name

            output_path = RESULT_DIR / (
                f"result_{Path(unique_name).stem}.jpg"
            )

            # Save the uploaded image.
            file.save(str(input_path))

            # Run lane detection with the selected settings.
            summary = detect_lanes(
                input_path,
                output_path,
                threshold=threshold,
                opacity=opacity
            )

            # Send both images to the frontend.
            input_url = url_for(
                "static",
                filename=f"uploads/{input_path.name}"
            )

            result_url = url_for(
                "static",
                filename=f"results/{output_path.name}"
            )

        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception(
                "An error occurred during lane detection."
            )
            error = (
                "Image processing failed. "
                "Please check the VS Code terminal."
            )

    return render_template(
        "index.html",
        error=error,
        result_url=result_url,
        input_url=input_url,
        summary=summary,
        threshold=threshold,
        opacity=opacity,
        lane_pixels=(
            summary["lane_pixels"] if summary else None
        )
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error=(
            "Image is too large. "
            "Please upload an image under 10 MB."
        ),
        result_url=None,
        input_url=None,
        summary=None,
        threshold=0.5,
        opacity=0.7,
        lane_pixels=None
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
