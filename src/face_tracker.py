import dlib

class FaceTracker():
    def __init__(self, face_detector, tracker = dlib.correlation_tracker()):
        self.face_detector = face_detector
        self.tracker = tracker
        self.is_tracking = False

    def track_faces(self, frame):
        if not self.is_tracking: 
            detected_faces = self.face_detector.detect_faces(frame)
            
            height, width, _ = frame.shape
            x1 = int(detected_faces[0][0] * width)
            y1 = int(detected_faces[0][1] * height)
            x2 = int(detected_faces[0][2] * width)
            y2 = int(detected_faces[0][3] * height)

            detected_faces_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

            self.tracker.start_track(image = frame, bounding_box = detected_faces_rect)
            return detected_faces
            
        else:
            tracking_quality = self.tracker.update(frame)

            if tracking_quality >= 8.75:
                tracked_position =  self.tracker.get_position()

                x1 = tracked_position.left()
                y1 = tracked_position.top()
                x2 = tracked_position.right()
                y2 = tracked_position.bottom()

                return np.array([[x1, y1, x2, y2]])
            
            else:
                self.is_tracking = False

