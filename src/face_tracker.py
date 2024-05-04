import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from sort.sort import *
import utils
import cv2


class FaceTracker:
    def __init__(
        self, face_detector, tracker=Sort(max_age=10, max_bbox_age=5)
    ):
        self.face_detector = face_detector
        self.tracker = tracker
        self.b = None

    def track_faces(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        detected_faces = self.face_detector.detect_faces(frame)
        if len(detected_faces) == 0:
            features = np.array([])
        else:
            detected_faces, features = detected_faces
        # detected_faces = utils.scale_bboxes(detected_faces, 1.2)

        frame = cv2.resize(frame, self.face_detector.det_res)
        if len(detected_faces) > 0:
            detections = utils.rescale_boxes(detected_faces, frame.shape[:2])
            detections = np.hstack(
                (detections, np.full((detections.shape[0], 1), 1))
            )  # add dummy confidences
        else:
            detections = np.empty((0, 5))

        # Update the tracker with the new frame detections and get the updated track information
        tracked_faces = self.tracker.update(detections)
        tracked_faces = tracked_faces[:, :4]  # throw away the confidence

        # Normalize the tracked coordinates back to [0, 1] range
        if tracked_faces.size > 0:
            height, width = frame.shape[:2]
            frame_res = np.array([height, width, height, width])
            tracked_faces /= frame_res
            tracked_faces = tracked_faces.astype(np.float32)
            return np.clip(tracked_faces, 0, 1), features
        else:
            return np.array([]), features
