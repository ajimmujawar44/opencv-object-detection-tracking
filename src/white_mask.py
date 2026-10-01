import cv2
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "videos" / "highway_short.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PORTRAIT CROP
# ============================================================

def crop_to_portrait(frame):

    height, width = frame.shape[:2]

    target_ratio = 9 / 16

    new_width = int(height * target_ratio)

    if new_width > width:
        return frame

    start_x = (width - new_width) // 2

    end_x = start_x + new_width

    return frame[:, start_x:end_x]


# ============================================================
# CREATE WHITE MASK
# ============================================================

def create_white_mask(frame):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # White color range
    lower_white = (0, 0, 180)

    upper_white = (180, 70, 255)

    mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# WHITE MASK PROCESSING
# ============================================================

def run_white_mask():

    print("\nStarting White Mask Processing...")
    print(f"Input video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        return

    print("Video opened successfully.")

    # Get FPS
    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30

    # Output file
    output_path = (
        OUTPUT_DIR /
        "white_mask_output.mp4"
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (450, 800)
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Video finished.")

            break

        # Crop to portrait
        frame = crop_to_portrait(
            frame
        )

        # Resize
        frame = cv2.resize(
            frame,
            (450, 800)
        )

        # Create mask
        mask = create_white_mask(
            frame
        )

        # Convert mask to BGR
        mask_display = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        # Save mask video
        writer.write(
            mask_display
        )

        # Display original
        cv2.imshow(
            "White Mask - Original",
            frame
        )

        # Display mask
        cv2.imshow(
            "White Mask",
            mask
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:

            break

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\nWhite mask output saved:")

    print(output_path)


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_white_mask()