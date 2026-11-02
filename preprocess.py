
import json
import cv2
import numpy as np
from pathlib import Path

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256

# TuSimple training dataset location
DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "archive"
    / "TUSimple"
    / "train_set"
)

# Preprocessed data will be saved inside the project
OUTPUT_ROOT = Path("dataset") / "processed"


def create_lane_mask(image_shape, lanes, h_samples):
    """Create a binary mask containing the lane markings."""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for lane in lanes:
        points = []

        for y, x in zip(h_samples, lane):
            if x < 0:
                continue

            if 0 <= x < width and 0 <= y < height:
                points.append((int(x), int(y)))

        if len(points) >= 2:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(
                mask,
                [points],
                isClosed=False,
                color=255,
                thickness=10
            )

    return mask


def process_dataset():
    image_output = OUTPUT_ROOT / "images"
    mask_output = OUTPUT_ROOT / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    label_files = [
        DATASET_ROOT / "label_data_0313.json",
        DATASET_ROOT / "label_data_0531.json",
        DATASET_ROOT / "label_data_0601.json",
    ]

    for label_file in label_files:
        if not label_file.is_file():
            raise FileNotFoundError(
                f"Label file not found: {label_file}"
            )

    processed = 0
    missing = 0

    for label_file in label_files:
        print(f"Reading labels: {label_file.name}")

        with label_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if "lanes" not in record or "h_samples" not in record:
                    continue

                raw_file = record.get("raw_file")

                if not raw_file:
                    continue

                image_path = DATASET_ROOT / Path(raw_file)

                if not image_path.is_file():
                    missing += 1
                    continue

                image = cv2.imread(str(image_path))

                if image is None:
                    missing += 1
                    continue

                mask = create_lane_mask(
                    image.shape,
                    record["lanes"],
                    record["h_samples"]
                )

                image = cv2.resize(
                    image,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                mask = cv2.resize(
                    mask,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )

                filename = Path(raw_file).with_suffix("").as_posix()
                filename = filename.replace("/", "_")

                image_path_out = image_output / f"{filename}.jpg"
                mask_path_out = mask_output / f"{filename}.png"

                if not cv2.imwrite(str(image_path_out), image):
                    raise OSError(f"Could not save image: {image_path_out}")

                if not cv2.imwrite(str(mask_path_out), mask):
                    raise OSError(f"Could not save mask: {mask_path_out}")

                processed += 1

    print("\nPreprocessing finished.")
    print(f"Processed image-mask pairs: {processed}")
    print(f"Missing or unreadable images: {missing}")
    print(f"Images saved to: {image_output.resolve()}")
    print(f"Masks saved to: {mask_output.resolve()}")

    if processed == 0:
        print("No images were processed. Check the dataset paths.")


if __name__ == "__main__":
    process_dataset()
