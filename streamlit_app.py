
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="LaneVision AI",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# Model Configuration
# -------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lane_unet_v2.keras"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
THRESHOLD = 0.2


# -------------------------------
# Load Trained U-Net Model
# -------------------------------
@st.cache_resource
def load_lane_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# -------------------------------
# Lane Detection
# -------------------------------
def detect_lanes(image):
    # Convert uploaded image to RGB
    original = np.array(image.convert("RGB"))

    # Resize image to model input dimensions
    resized = cv2.resize(
        original,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # Normalize pixel values
    input_image = resized.astype(np.float32) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(
        input_image,
        axis=0
    )

    # Predict lane probabilities
    model = load_lane_model()

    prediction = model.predict(
        input_image,
        verbose=0
    )[0, :, :, 0]

    # Create binary lane mask
    mask = (
        prediction > THRESHOLD
    ).astype(np.uint8) * 255

    # Resize mask to original image dimensions
    mask = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Highlight predicted lane pixels in green
    overlay = original.copy()
    overlay[mask > 0] = (0, 255, 0)

    # Blend original image and overlay
    result = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )

    return original, mask, result


# -------------------------------
# Streamlit User Interface
# -------------------------------
st.title("LaneVision AI")
st.subheader(
    "U-Net-Based Lane Detection for Autonomous Driving"
)

st.write(
    "Upload a road image to visualize the lanes "
    "predicted by the trained U-Net model."
)

st.info(
    "This is an image-based lane detection prototype. "
    "Predicted markings are model outputs, not guaranteed "
    "to be accurate driving boundaries."
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting lanes..."):
            original, mask, result = detect_lanes(image)

        st.success("Lane detection prediction completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original Road Image",
                use_container_width=True
            )

        with col2:
            st.image(
                result,
                caption="Predicted Lane Overlay",
                use_container_width=True
            )

        st.subheader("Predicted Lane Mask")

        st.image(
            mask,
            caption="White pixels indicate predicted lane regions",
            use_container_width=True,
            clamp=True
        )

        # Prepare the result as a downloadable PNG
        result_buffer = io.BytesIO()

        Image.fromarray(result).save(
            result_buffer,
            format="PNG"
        )

        st.download_button(
            label="Download Lane Detection Result",
            data=result_buffer.getvalue(),
            file_name="lane_prediction.png",
            mime="image/png"
        )

    except Exception as error:
        st.error(
            f"Could not process the image: {error}"
        )

        st.caption(
            "Check that the model file exists and that "
            "the required Python libraries are installed."
        )
