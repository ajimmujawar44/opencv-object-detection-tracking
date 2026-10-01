import cv2
from pathlib import Path

from src.tracker import CentroidTracker, draw_tracking
from src.white_mask import run_white_mask


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_PATH = PROJECT_ROOT / "videos" / "highway_short.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PORTRAIT CROP
# ============================================================

def crop_to_portrait(frame):
    """
    Crop landscape video into a 9:16 portrait frame.
    """

    height, width = frame.shape[:2]

    target_ratio = 9 / 16
    new_width = int(height * target_ratio)

    if new_width > width:
        return frame

    start_x = (width - new_width) // 2
    end_x = start_x + new_width

    return frame[:, start_x:end_x]


# ============================================================
# CREATE VIDEO WRITER
# ============================================================

def create_video_writer(output_path, fps=30):
    """
    Create a VideoWriter for saving processed video.
    """

    width = 450
    height = 800

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    return writer


# ============================================================
# OBJECT DETECTION
# ============================================================

def run_detection():

    print("\nStarting Object Detection...")
    print(f"Input video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Video opened successfully.")

    # Get original FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    output_path = OUTPUT_DIR / "object_detection_output.mp4"

    writer = create_video_writer(output_path, fps)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        # Crop landscape → portrait
        frame = crop_to_portrait(frame)

        # Resize
        frame = cv2.resize(frame, (450, 800))

        # Background subtraction
        mask = background_subtractor.apply(frame)

        # Remove shadows
        _, mask = cv2.threshold(
            mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # Morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < 15 or h < 15:
                continue

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Center point
            center_x = x + w // 2
            center_y = y + h // 2

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

        # Save processed frame
        writer.write(frame)

        # Display
        cv2.imshow("Object Detection", frame)
        cv2.imshow("Detection Mask", mask)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nDetection output saved:")
    print(output_path)


# ============================================================
# OBJECT TRACKING
# ============================================================

def run_tracking():

    print("\nStarting Object Tracking...")
    print(f"Input video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Video opened successfully.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    output_path = OUTPUT_DIR / "object_tracking_output.mp4"

    writer = create_video_writer(output_path, fps)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    tracker = CentroidTracker(
        max_disappeared=15,
        max_distance=60
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        # ----------------------------------------------------
        # Crop landscape video to portrait
        # ----------------------------------------------------

        frame = crop_to_portrait(frame)

        frame = cv2.resize(
            frame,
            (450, 800)
        )

        # ----------------------------------------------------
        # Background subtraction
        # ----------------------------------------------------

        mask = background_subtractor.apply(frame)

        _, mask = cv2.threshold(
            mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # ----------------------------------------------------
        # Morphological operations
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        rects = []

        # IMPORTANT:
        # All continue statements are INSIDE this loop.

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore very small objects
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very small objects
            if w < 20 or h < 20:
                continue

            # Ignore extremely large objects
            if w > 250 or h > 300:
                continue

            # Calculate aspect ratio
            aspect_ratio = w / float(h)

            # Ignore very thin objects
            if aspect_ratio < 0.25:
                continue

            # Ignore extremely wide objects
            if aspect_ratio > 4.0:
                continue

            rects.append(
                (x, y, w, h)
            )

        # ----------------------------------------------------
        # Update tracker
        # ----------------------------------------------------

        objects = tracker.update(rects)

        # ----------------------------------------------------
        # Draw bounding boxes
        # ----------------------------------------------------

        for object_id, centroid in objects.items():

            center_x, center_y = centroid

            closest_rect = None

            smallest_distance = float("inf")

            for x, y, w, h in rects:

                rect_center_x = x + w // 2

                rect_center_y = y + h // 2

                distance = (
                    (center_x - rect_center_x) ** 2
                    +
                    (center_y - rect_center_y) ** 2
                ) ** 0.5

                if distance < smallest_distance:

                    smallest_distance = distance

                    closest_rect = (
                        x,
                        y,
                        w,
                        h
                    )

            if closest_rect is not None:

                x, y, w, h = closest_rect

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        # ----------------------------------------------------
        # Draw object IDs
        # ----------------------------------------------------

        frame = draw_tracking(
            frame,
            objects
        )

        # ----------------------------------------------------
        # Save processed frame
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Object Tracking",
            frame
        )

        cv2.imshow(
            "Tracking Mask",
            mask
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            break

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\nTracking output saved:")
    print(output_path)

# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("\n==========================================")
    print(" HIGHWAY VEHICLE DETECTION PROJECT")
    print("==========================================")
    print()
    print("1. Object Detection")
    print("2. Object Tracking")
    print("3. White Mask")
    print("4. Exit")
    print()

    choice = input("Enter your choice (1-4): ")

    if choice == "1":


        run_detection()

    elif choice == "2":

        run_tracking()

    elif choice == "3":

        run_white_mask()

    elif choice == "4":

        print("\nExiting program...")

    else:

        print("\nInvalid choice. Please select 1, 2, 3, or 4.")

# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()