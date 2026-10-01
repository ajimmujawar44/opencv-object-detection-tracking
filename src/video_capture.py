
import cv2
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the input video
VIDEO_PATH = PROJECT_ROOT / "videos" / "highway_short.mp4"


# 2. OPEN VIDEO

cap = cv2.VideoCapture(str(VIDEO_PATH))

# Check whether the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file:\n{VIDEO_PATH}")
    raise SystemExit(1)

print("Video opened successfully.")
print("Press 'q' or 'Esc' to exit.")


# 3. READ VIDEO FRAME BY FRAME

while True:

    # Read one frame from the video
    ret, frame = cap.read()

    # If no frame is returned, the video has ended
    # or there was a problem reading it.
    if not ret:
        print("End of video or unable to read frame.")
        break

    # --------------------------------------------------------
    # Resize frame for display
    # --------------------------------------------------------
    # This keeps large videos from occupying the entire screen.
    display_frame = cv2.resize(frame, (900, 600))

    # Display the current frame
    cv2.imshow("Highway Video", display_frame)

    # Wait 30 milliseconds for a keyboard event
    key = cv2.waitKey(30) & 0xFF

    # Press 'q' or ESC to exit
    if key == ord("q") or key == 27:
        print("Video stopped by user.")
        break



# 4. RELEASE RESOURCES

cap.release()
cv2.destroyAllWindows()

print("Video resources released.")

