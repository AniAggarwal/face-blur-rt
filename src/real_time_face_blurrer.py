from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy import ndarray
from pathlib import Path

import utils
from video_input_handler import VideoInputHandler
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker
from blurrer import Blurrer, BlurringMethod, BlurringShape
from performance_settings import PerformanceSettings


class RealTimeFaceBlurrer(ABC):

    def __init__(
        self,
        video_source: int | str,
        face_recognizer: FaceRecognizer,
        face_detector: FaceDetector,
        face_tracker: FaceTracker,
        blurring_method: BlurringMethod,
        blurring_shape: BlurringShape,
        performance_settings: PerformanceSettings,
        use_face_tracker: bool,
    ):
        self.video_input = VideoInputHandler(video_source)
        self.face_recognizer = face_recognizer
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.blurrer = Blurrer(blurring_method, blurring_shape)
        self.performance_settings = performance_settings
        self.use_face_tracker = use_face_tracker

    @abstractmethod
    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""
        pass


class RealTimeFaceBlurrerByFrame(RealTimeFaceBlurrer):

    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""

        tick_meter = cv2.TickMeter()
        tick_meter.reset()

        # cv2.startWindowThread()

        while True:
            tick_meter.start()

            ret, frame = self.video_input.get_frame()
            if not ret:
                print("No more frames.")
                break

            # resize to target res
            frame = cv2.resize(frame, self.performance_settings.resolution)

            faces = None
            if self.use_face_tracker:
                faces = self.face_tracker.track_faces(frame)
                recognition_faces = faces[1]
            else:
                print("not using tracker")
                faces = self.face_detector.detect_faces(frame)
                num, recognition_faces = self.face_detector.detector.detect(
                    cv2.resize(frame, self.face_detector.det_res)
                )
            # rescale bboxes to original frame size
            if len(faces) != 0:
                faces = utils.rescale_boxes(
                    faces[0], self.performance_settings.resolution
                )

            print(f"tracked {len(faces)} faces.")
            if recognition_faces is None:
                recognition_faces = []
            recognition_faces = [face for face in recognition_faces if face is not None]

            recognized_dict = self.face_recognizer.recognize_faces(
                frame, recognition_faces
            )

            unknown_faces = []
            for i, face in enumerate(faces):
                if i in recognized_dict:
                    print("Recognized face:", recognized_dict[i])
                    # Apply blurring if face is unknown
                    if recognized_dict[i] is None:
                        unknown_faces.append(face)
                    if self.performance_settings.enable_labels:
                        (w, h), _ = cv2.getTextSize(
                            recognized_dict[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        frame = cv2.rectangle(
                            frame,
                            (face[0], face[1]),
                            (face[0] + w, face[1] - h - 10),
                            (255, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            recognized_dict[i],
                            (face[0], face[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
            # Apply blurring to unrecognized faces
            frame = self.blurrer.apply_blur(frame, unknown_faces)

            tick_meter.stop()
            if self.performance_settings.fps_counter:
                cv2.putText(
                    frame,
                    f"FPS: {tick_meter.getFPS():.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                # print(f"FPS: {tick_meter.getFPS():.2f}")

            # Show the processed frame.
            cv2.imshow("Real-Time Face Blurring, Frame over Frame", frame)

            # Break loop with 'q' key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_input.release()
        cv2.destroyAllWindows()
