from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy import ndarray
from pathlib import Path

import utils
from video_input_handler import VideoInputHandler
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from blurrer import Blurrer, BlurringMethod, BlurringShape
from performance_settings import PerformanceSettings


class RealTimeFaceBlurrer(ABC):
    def __init__(
        self,
        video_source: int | str,
        face_detector: FaceDetector,
        face_recognizer: FaceRecognizer,
        blurring_method: BlurringMethod,
        blurring_shape: BlurringShape,
        performance_settings: PerformanceSettings,
    ):
        self.video_input = VideoInputHandler(video_source)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.blurrer = Blurrer(blurring_method, blurring_shape)
        self.performance_settings = performance_settings

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

            # Detect and recognize faces
            detected_faces = self.face_detector.detect_faces(frame)

            # rescale bboxes to original frame size
            detected_faces = utils.rescale_boxes(
                detected_faces, self.performance_settings.resolution
            )

            print(f"Detected {len(detected_faces)} faces.")

            # TODO: uncomment this when face_recognizer is implemented
            # and not super slow

            # recognized_dict = self.face_recognizer.recognize_faces(
            #     frame, detected_faces
            # )
            #
            # for i, face in enumerate(detected_faces):
            #     if i in recognized_dict:
            #         print("Recognized face:", recognized_dict[i])
            #         # Apply blurring to recognized faces
            #         frame = self.blurrer.apply_blur(frame, face.reshape(1, -1))
            #     else:
            #         print("Unrecognized face")

            # Apply blurring to unrecognized faces
            frame = self.blurrer.apply_blur(frame, detected_faces)

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
