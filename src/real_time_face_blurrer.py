import cv2
import numpy as np


class RealTimeFaceBlurrer:
    def __init__(
        self,
        video_source,
        face_detection_model,
        face_recognition_model,
        blurring_method="bounding_box",
        performance_mode="high_accuracy",
    ):
        self.video_input = VideoInputHandler(video_source)
        self.face_detector = FaceDetector(face_detection_model)
        self.face_recognizer = FaceRecognizer(face_recognition_model)
        self.blurrer = Blurrer(blurring_method)
        self.performance_settings = PerformanceSettings(performance_mode)
        self.temporal_tracker = TemporalFaceTracker()

    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""
        while True:
            frame = self.video_input.get_frame()
            if frame is None:
                break

            # Detect and recognize faces.
            detected_faces = self.face_detector.detect_faces(frame)
            recognized_faces = self.face_recognizer.recognize_faces(
                frame, detected_faces
            )

            # Apply blurring to unrecognized faces.
            frame = self.blurrer.apply_blur(frame, detected_faces)

            # Show the processed frame.
            cv2.imshow("Real-Time Face Blurring", frame)

            # Break loop with 'q' key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_input.release()
        cv2.destroyAllWindows()
