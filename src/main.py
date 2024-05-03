import cv2
from pathlib import Path

from real_time_face_blurrer import RealTimeFaceBlurrerByFrame
from blurrer import BlurringMethod, BlurringShape
from face_detector import SCRFDDetector, YuNetDetector
from face_recognizer import FaceRecognizer, SFRecognizer
from face_tracker import FaceTracker
from performance_settings import PerformanceSettings


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
    # video_source = 0  # Webcam source
    # video_source = str(Path("./data/demos/one-person.webm").resolve())
    # video_source = str(Path("./data/demos/GeoVision.webm").resolve())
    # video_source = str(Path("./data/demos/multi.webm").resolve())

    video_source = str(
        Path("./data/demos/irl-stream-long-30fps.mp4").resolve()
    )
    # set to None to not output to CSV
    output_csv = Path(
        "./data/demos/irl-stream-long-30fps-ours-detection-only.csv"
    ).resolve()
    # output_csv = None

    if output_csv is not None:
        if output_csv.exists():
            if (
                input("Output CSV file already exists. Overwrite? (y/n): ")
                == "y"
            ):
                output_csv.unlink()
            else:
                print("Appending to existing CSV file.")
        print("Will output to CSV file:", output_csv)

        with open(output_csv, "w") as f:
            f.write('"frame_num","time_elapsed","bboxes"\n')

    blur_method = BlurringMethod.LINE
    # blur_method = BlurringMethod.BLACK
    # blur_method = BlurringMethod.GAUSSIAN

    blur_shape = BlurringShape.SQUARE
    # blur_shape = BlurringShape.CIRCLE
    # view_camera(0)

    detector_path = Path(
        "./models/face_detection_yunet_2023mar.onnx"
    ).resolve()
    recognizer_path = Path(
        "./models/face_recognition_sface_2021dec.onnx"
    ).resolve()
    known_faces_path = Path("./data/known-faces").resolve()

    face_detection_model = YuNetDetector(detector_path)
    face_recognition_model = SFRecognizer(
        recognizer_path, face_detection_model, known_faces_path
    )
    face_tracker = FaceTracker(face_detection_model)

    use_face_tracker = False

    performance_settings = PerformanceSettings(
        (640, 480), 30, False, False, False
    )

    real_time_blurrer = RealTimeFaceBlurrerByFrame(
        video_source,
        face_recognition_model,
        face_detection_model,
        face_tracker,
        blur_method,
        blur_shape,
        performance_settings,
        use_face_tracker,
        output_csv,
    )
    real_time_blurrer.process_stream()
