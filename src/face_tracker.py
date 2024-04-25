import dlib
import numpy as np
import utils

class FaceTracker():
    def __init__(self, face_detector, tracker = dlib.correlation_tracker()):
        self.face_detector = face_detector
        self.tracker = tracker
        self.is_tracking = False

    def track_faces(self, frame):
        if not self.is_tracking:
            detected_faces = self.face_detector.detect_faces(frame)
            if len(detected_faces) == 0:
                return np.array([])

            # detect_faces returns normalized coordinates, so rescale them 
            # to match the dimensions of frame
            detected_faces_rescaled = utils.rescale_boxes(detected_faces, frame.shape[:2])
            x1, y1, x2, y2 = detected_faces_rescaled[0]

            detected_faces_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2) 

            self.tracker.start_track(image = frame, bounding_box = detected_faces_rect)
            self.is_tracking = True
            return detected_faces
            
        else:
            tracking_quality = self.tracker.update(frame)

            if tracking_quality >= 7:
                frame_res = (frame.shape[0], frame.shape[1])

                tracked_position =  self.tracker.get_position()

                tl_corner = tracked_position.tl_corner()
                br_corner = tracked_position.br_corner()

                x1 = tl_corner.x
                y1 = tl_corner.y
                x2 = br_corner.x
                y2 = br_corner.y

                tracked_faces = np.array([[x1, y1, x2, y2]])

                tracked_faces /= np.array([*frame_res, *frame_res])

                return tracked_faces.astype(np.float32)
            
            else:
                self.is_tracking = False
                return self.face_detector.detect_faces(frame)

