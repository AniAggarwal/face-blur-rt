import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from sort.sort import *  # Assuming you have the SORT algorithm installed
import utils

class FaceTracker():
    def __init__(self, face_detector, tracker=Sort(max_age=20, min_hits=1)):
        self.face_detector = face_detector
        self.tracker = tracker
        self.b = None

    def track_faces(self, frame):
        detected_faces = self.face_detector.detect_faces(frame)
        # detected_faces = utils.scale_bboxes(detected_faces, 1.2)

        # Convert detections to the format required by SORT: [x1, y1, x2, y2, confidence]
        if len(detected_faces) > 0:
            detections = utils.rescale_boxes(detected_faces, frame.shape[:2])
            detections = np.hstack((detections, np.full((detections.shape[0], 1), 0.5)))  # add dummy confidences
        else:
            detections = np.empty((0, 5))

        # Update the tracker with the new frame detections and get the updated track information
        
        tracked_faces = self.tracker.update(detections)
        print(tracked_faces[:, 4:])
        tracked_faces = tracked_faces[:, :4]

        # Normalize the tracked coordinates back to [0, 1] range
        if tracked_faces.size > 0:
            height, width = frame.shape[:2]
            frame_res = np.array([height, width, height, width])
            tracked_faces /= frame_res  # Normalize coordinates
            return tracked_faces.astype(np.float32)
        else:
            return np.array([])  # Return empty array if no faces tracked

# Note: You will need to ensure that utils.rescale_boxes and the detect_faces method handle coordinates correctly.
