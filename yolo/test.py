import logging
from pathlib import Path
import sys

import cv2

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import configure_logging


logger = logging.getLogger("src.yolo.test")


def main():
    # Load the YOLO model.
    model = YOLO(r"D:\private\multi-camera-system\yolo\results\detect\runs\train\person_vehicle\weights\best.pt")
    results = model.predict(r"C:\Users\luann\OneDrive\Pictures\test.jpg", conf=0.5)
    logger.info("Inference completed: frames=%d detections=%d", len(results),
                sum(len(result.boxes) for result in results if result.boxes is not None))
    results[0].plot()


if __name__ == "__main__":
    with configure_logging():
        main()

#
# # Open the video file
# video_path = r"D:\private\Videos\traffic_pl_2.mp4"
#
# res = model.track(video_path, show=True, tracker="bytetrack.yaml")
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLO inference on the frame
#         results = model(frame, conf=0.5)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLO Inference", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()



# import cv2
#
# # Open the video file
# cap = cv2.VideoCapture(r"D:\private\Videos\traffic_pl_2.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # Check if the video was opened successfully
# if not cap.isOpened():
# else:
#
# while True:
#     # Read the first frame to confirm reading
#     ret, frame = cap.read()
#
#     if ret:
#         # Display the frame using imshow
#         cv2.imshow("Camera", frame)
#
#     delay = int(1000 / fps)  # Calculate the delay in milliseconds
#     if cv2.waitKey(delay) == 30:  # Exit if the 'Esc' key is pressed
#         break
#
# # Release the video capture object
#
# cap.release()
# cv2.destroyAllWindows()


# import yaml
#
# with open(r"D:\private\multi-camera-system\configs\cameras.yaml", "r", encoding="utf-8") as f:
#     configs = yaml.safe_load(f)
