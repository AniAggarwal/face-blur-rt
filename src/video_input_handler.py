import cv2
import numpy as np


class VideoInputHandler:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        """Capture a single frame from the video source."""
        ret, frame = self.cap.read()
        return frame

    def release(self):
        """Release the video capture object."""
        self.cap.release()
