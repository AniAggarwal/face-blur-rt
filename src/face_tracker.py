import dlib
import numpy as np
import utils
import copy
import cv2

class FaceTracker():
    def __init__(self, face_detector, tracker_model = dlib.correlation_tracker()):
        self.face_detector = face_detector
        self.tracker_model = tracker_model
        self.trackers = []

    def track_faces(self, frame, in_detection_mode: bool):
        width, height = frame.shape[:2]
        tracked_faces = []

        new_trackers = []
        for tracker in self.trackers:
            tracking_quality = tracker.update(frame)

            if tracking_quality >= 12:

                tracked_rect =  tracker.get_position()
                bbox = utils.dlib_rect_to_bbox(tracked_rect)

                tracked_faces.append(bbox)
                new_trackers.append(tracker)
            else:
                print('lost a face')
                in_detection_mode = True

        # new_trackers only contains trackers that are still tracking a face in the current frame
        self.trackers = new_trackers
        

        
        if in_detection_mode:
            detected_faces = self.face_detector.detect_faces(frame)
            detected_faces_rescaled = utils.rescale_boxes(detected_faces, frame.shape[:2])
            # detected_faces = utils.scale_bboxes(detected_faces, 1.2)

            # Discard detected faces if they overlap with a tracked face
            detected_faces_filtered_rescaled = list(detected_faces_rescaled)
            for detector_bbox in detected_faces_rescaled:
                for tracker in self.trackers:
                    tracker_bbox = utils.dlib_rect_to_bbox(tracker.get_position())
                    if utils.bbox_overlap(detector_bbox, tracker_bbox):
                        print('hi')
                        detected_faces_filtered_rescaled.pop(0)

            # detect_faces returns normalized coordinates, so rescale them 
            # to match the dimensions of frame
            for face in detected_faces_filtered_rescaled:
                detected_faces_rect = utils.bbox_to_dlib_rect(face)

                tracker = dlib.correlation_tracker()
                tracker.start_track(image = frame, bounding_box = detected_faces_rect)
                self.trackers.append(tracker)

            print("detecting")
            tracked_faces.extend(detected_faces_filtered_rescaled)

        print(f'tracking {len(tracked_faces)} faces')
        if len(tracked_faces) == 0:
            return np.array([])
        
        tracked_faces = tracked_faces / np.array([width, height, width, height])
        tracked_faces = np.array(tracked_faces)
        return tracked_faces.astype(np.float32)
            
        
