 
from pathlib import Path
import argparse

import cv2
import numpy as np
import tensorflow as tf


# Image dimensions expected by the U-Net model
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

# File locations
MODEL_PATH = Path("models") / "lane_unet.keras"
OUTPUT_DIR = Path("outputs")


def predict_lanes(image_path):
    image_path = Path(image_path)

    # Check whether the trained model exists
    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        print("Train the U-Net model before making predictions.")
        return

    # Read the original road image
    original = cv2.imread(str(image_path))

    if original is None:
        print(f"Could not read image: {image_path}")
        return

    # Resize image to the model's expected dimensions
    resized = cv2.resize(
        original,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # Convert BGR to RGB and normalize
    rgb_image = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    input_image = rgb_image.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Load the trained model
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    # Predict lane probability for every pixel
    prediction = model.predict(
        input_image,
        verbose=0
    )[0, :, :, 0]

    print("\n--- Prediction Statistics ---")
    print("Prediction minimum:", float(prediction.min()))
    print("Prediction maximum:", float(prediction.max()))
    print("Prediction mean:", float(prediction.mean()))

    print("Pixels above 0.05:", int(np.sum(prediction > 0.05)))
    print("Pixels above 0.10:", int(np.sum(prediction > 0.10)))
    print("Pixels above 0.20:", int(np.sum(prediction > 0.20)))
    print("Pixels above 0.30:", int(np.sum(prediction > 0.30)))
    print("Pixels above 0.40:", int(np.sum(prediction > 0.40)))

    # Create the binary lane mask
    threshold = 0.2
    lane_mask = (
        prediction > threshold
    ).astype(np.uint8) * 255

    print("Threshold:", threshold)
    print("Predicted lane pixels:", int(np.count_nonzero(lane_mask)))
    print("Total pixels:", lane_mask.size)

    # Resize the mask to the original image dimensions
    lane_mask = cv2.resize(
        lane_mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Create the output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the raw predicted mask
    mask_path = OUTPUT_DIR / "lane_mask_debug.png"

    if cv2.imwrite(str(mask_path), lane_mask):
        print(f"Raw mask saved to: {mask_path.resolve()}")
    else:
        print("Warning: Could not save the raw mask.")

    # Overlay predicted lane pixels in green
    overlay = original.copy()
    overlay[lane_mask > 0] = (0, 255, 0)

    result = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )

    # Save the overlay result
    output_path = OUTPUT_DIR / "lane_prediction.jpg"

    if cv2.imwrite(str(output_path), result):
        print("Lane prediction completed.")
        print(f"Overlay saved to: {output_path.resolve()}")
    else:
        print("Could not save the overlay image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict road lanes using U-Net."
    )

    parser.add_argument(
        "image",
        help="Path to the road image"
    )

    args = parser.parse_args()
    predict_lanes(args.image)
