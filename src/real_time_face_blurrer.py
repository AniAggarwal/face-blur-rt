from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy import ndarray
from pathlib import Path
import torch

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
        output_csv: Path | None = None,
    ):
        self.video_input = VideoInputHandler(video_source)
        self.face_recognizer = face_recognizer
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.blurrer = Blurrer(blurring_method, blurring_shape)
        self.performance_settings = performance_settings
        self.use_face_tracker = use_face_tracker
        self.output_csv = output_csv

    @abstractmethod
    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""
        pass


class RealTimeFaceBlurrerByFrame(RealTimeFaceBlurrer):
    def process_stream(self):
        """Process the video stream and apply face blurring in real-time."""

        frame_num = 0
        tick_meter = cv2.TickMeter()
        tick_meter.reset()

        # cv2.startWindowThread()

        while True:
            time_start = torch.cuda.Event(enable_timing=True)
            time_start.record()
            tick_meter.start()

            ret, frame = self.video_input.get_frame()
            if not ret:
                print("No more frames.")
                break

            # resize to target res
            frame = cv2.resize(frame, self.performance_settings.resolution)

            bboxes = None
            if self.use_face_tracker:
                bboxes = self.face_tracker.track_faces(frame)
            else:
                print("not using tracker")
                bboxes = self.face_detector.detect_faces(frame)

            # rescale bboxes to original frame size
            faces = utils.rescale_boxes(
                bboxes, self.performance_settings.resolution
            )

            # print(f"tracked {len(tracked_faces)} faces.")

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
            if self.performance_settings.apply_blur:
                frame = self.blurrer.apply_blur(frame, faces)

            tick_meter.stop()
            time_end = torch.cuda.Event(enable_timing=True)
            time_end.record()
            torch.cuda.synchronize()
            time_elapsed = time_start.elapsed_time(time_end)

            # output to csv if specified
            if self.output_csv is not None:
                utils.bboxes_to_csv(
                    self.output_csv, bboxes, frame_num, time_elapsed
                )

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
            if self.performance_settings.display_video:
                cv2.imshow("Real-Time Face Blurring, Frame over Frame", frame)

            frame_num += 1
            # Break loop with 'q' key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_input.release()
        cv2.destroyAllWindows()
