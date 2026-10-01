import cv2
from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "videos" / "highway_short.mp4"


# ============================================================
# 2. PORTRAIT CROP FUNCTION
# ============================================================

def crop_to_portrait(frame):
    """
    Convert a landscape frame into a 9:16 portrait frame.

    The center portion of the original video is cropped.
    Nothing is stretched.
    """

    height, width = frame.shape[:2]

    # Desired portrait ratio = 9:16
    target_ratio = 9 / 16

    # Calculate required width
    new_width = int(height * target_ratio)

    # Center crop
    x_start = (width - new_width) // 2
    x_end = x_start + new_width

    portrait = frame[:, x_start:x_end]

    return portrait


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():

    print("ERROR: Could not open video.")
    print(f"Video path: {VIDEO_PATH}")

    raise SystemExit(1)


print("==========================================")
print("Vehicle Detection Started")
print("==========================================")
print(f"Video: {VIDEO_PATH}")
print()
print("Press Q or ESC to exit.")
print()


# ============================================================
# 4. BACKGROUND SUBTRACTOR
# ============================================================

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)


# ============================================================
# 5. WINDOW SETTINGS
# ============================================================

WINDOW_WIDTH = 450
WINDOW_HEIGHT = 800


cv2.namedWindow(
    "Vehicle Detection",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Clean Mask",
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    "Vehicle Detection",
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)

cv2.resizeWindow(
    "Clean Mask",
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)


# Position windows

cv2.moveWindow(
    "Vehicle Detection",
    20,
    30
)

cv2.moveWindow(
    "Clean Mask",
    500,
    30
)


# ============================================================
# 6. PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("End of video.")

        break


    # ========================================================
    # 7. CONVERT LANDSCAPE → PORTRAIT
    # ========================================================

    frame = crop_to_portrait(frame)


    # ========================================================
    # 8. RESIZE PORTRAIT VIDEO
    # ========================================================

    frame = cv2.resize(
        frame,
        (450, 800),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # 9. DEFINE ROI
    # ========================================================

    # Since the video is now portrait:
    #
    # Width  = 450
    # Height = 800
    #
    # We process most of the visible road.

    roi = frame[100:780, 50:400]


    # ========================================================
    # 10. BACKGROUND SUBTRACTION
    # ========================================================

    mask = background_subtractor.apply(roi)


    # ========================================================
    # 11. REMOVE SHADOWS
    # ========================================================

    _, mask = cv2.threshold(
        mask,
        200,
        255,
        cv2.THRESH_BINARY
    )


    # ========================================================
    # 12. MORPHOLOGICAL FILTERING
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )


    # Remove small noise

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # Connect nearby regions

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # 13. FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # 14. OBJECT DETECTION
    # ========================================================

    for contour in contours:

        # Object area

        area = cv2.contourArea(contour)


        # Ignore tiny objects

        if area < 300:
            continue


        # Bounding rectangle

        x, y, w, h = cv2.boundingRect(contour)


        # ----------------------------------------------------
        # Object size filtering
        # ----------------------------------------------------

        if w < 15 or h < 15:
            continue


        if w > 200 or h > 250:
            continue


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            roi,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ====================================================
        # CENTER POINT
        # ====================================================

        center_x = x + w // 2
        center_y = y + h // 2


        cv2.circle(
            roi,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )


        # ====================================================
        # OBJECT SIZE
        # ====================================================

        cv2.putText(
            roi,
            f"{w}x{h}",
            (
                x,
                max(y - 7, 15)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # 15. PUT ROI BACK
    # ========================================================

    frame[
        100:780,
        50:400
    ] = roi


    # ========================================================
    # 16. DRAW ROI BORDER
    # ========================================================

    cv2.rectangle(
        frame,
        (50, 100),
        (400, 779),
        (255, 0, 0),
        2
    )


    # ========================================================
    # 17. CREATE PORTRAIT MASK
    # ========================================================

    # Resize mask to same display dimensions

    display_mask = cv2.resize(
        mask,
        (450, 680),
        interpolation=cv2.INTER_NEAREST
    )


    # Create black portrait background

    full_mask = cv2.copyMakeBorder(
        display_mask,
        100,
        20,
        50,
        50,
        cv2.BORDER_CONSTANT,
        value=0
    )


    # ========================================================
    # 18. DISPLAY
    # ========================================================

    cv2.imshow(
        "Vehicle Detection",
        frame
    )

    cv2.imshow(
        "Clean Mask",
        full_mask
    )


    # ========================================================
    # 19. KEYBOARD
    # ========================================================

    key = cv2.waitKey(30) & 0xFF


    if key == ord("q") or key == 27:

        print("Stopped by user.")

        break


# ============================================================
# 20. RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("==========================================")
print("Program Finished")
print("==========================================")