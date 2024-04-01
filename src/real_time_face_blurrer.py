from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy import ndarray
from pathlib import Path

from video_input_handler import VideoInputHandler
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from blurrer import Blurrer
from performance_settings import PerformanceSettings


class RealTimeFaceBlurrer(ABC):
    def __init__(
        self,
        video_source: int | str,
        face_detector: FaceDetector,
        face_recognizer: FaceRecognizer,
        blurring_method: str = "bounding_box",
        resolution: tuple[int, int] = (640, 480),
        target_fps: int = 24,
    ):
        self.video_input = VideoInputHandler(video_source)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.blurrer = Blurrer(blurring_method)
        self.performance_settings = PerformanceSettings(resolution, target_fps)

    @abstractmethod
    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""
        pass

    def rescale_boxes(self, boxes: ndarray, original_size: tuple[int, int]):
        """
        Rescale bounding box coordinates from fractional representation to absolute pixel values.
        Will skip processing if the input boxes are empty.

        Args:
            boxes: np.array of shape (n, 4) with each row being (x1, y1, x2, y2) as fractions of frame.
            original_size: tuple (width, height) of the original frame size in pixels.

        Returns:
        - scaled_boxes: np.array of shape (n, 4) with each row being (x1, y1, x2, y2) in absolute pixels.
        """
        if boxes.size == 0:
            return boxes

        # Extract the original frame dimensions
        width, height = original_size

        # Scale the coordinates
        scaled_boxes = boxes * np.array([width, height, width, height])

        return scaled_boxes.astype(int)

    def draw_bboxes(self, frame: ndarray, bboxes: ndarray) -> None:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


class RealTimeFaceBlurrerByFrame(RealTimeFaceBlurrer):
    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""
        while True:
            frame = self.video_input.get_frame()
            if frame is None:
                break

            # resize to target res
            cv2.resize(frame, self.performance_settings.resolution, dst=frame)

            # Detect and recognize faces
            detected_faces = self.face_detector.detect_faces(frame)

            # rescale bboxes to original frame size
            detected_faces = self.rescale_boxes(
                detected_faces, self.performance_settings.resolution
            )

            # Draw bounding boxes around detected faces
            self.draw_bboxes(frame, detected_faces)

            recognized_faces = self.face_recognizer.recognize_faces(
                frame, detected_faces
            )

            # TODO: Implement face recognition

            # Apply blurring to unrecognized faces
            frame = self.blurrer.apply_blur(frame, detected_faces)

            # Show the processed frame.
            cv2.imshow("Real-Time Face Blurring, Frame over Frame", frame)

            # Break loop with 'q' key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_input.release()
        cv2.destroyAllWindows()
