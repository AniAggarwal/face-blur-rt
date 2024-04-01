import cv2
from pathlib import Path

from real_time_face_blurrer import RealTimeFaceBlurrerByFrame
from face_detector import SCRFDDetector, YuNetDetector
from face_recognizer import FaceRecognizer


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
    # Example usage
    video_source = 0  # Webcam source
    # view_camera(0)

    # detector_path = Path("./models/SCRFD_10G.pth").resolve()
    # face_detection_model = SCRFDDetector(detector_path)

    detector_path = Path(
        "./models/face_detection_yunet_2023mar.onnx"
    ).resolve()
    face_detection_model = YuNetDetector(detector_path)
    face_recognition_model = FaceRecognizer("")

    real_time_blurrer = RealTimeFaceBlurrerByFrame(
        video_source, face_detection_model, face_recognition_model
    )
    real_time_blurrer.process_stream()
