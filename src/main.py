import cv2
import numpy as np

from real_time_face_blurrer import RealTimeFaceBlurrer


def view_camera(video_source: int, window_name: str = "Camera") -> None:
    """View the camera feed in a window."""
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    view_camera(0)
    # Example usage
    # video_source = 0  # Webcam source
    # face_detection_model = "path/to/face/detection/model"
    # face_recognition_model = "path/to/face/recognition/model"
    # real_time_blurrer = RealTimeFaceBlurrer(
    #     video_source, face_detection_model, face_recognition_model
    # )
    # real_time_blurrer.process_stream()
