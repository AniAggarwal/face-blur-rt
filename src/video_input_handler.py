import cv2
import numpy as np
from pathlib import Path
from numpy import ndarray


class VideoInputHandler:
    def __init__(self, source: int | Path | str = 0) -> None:
        if isinstance(source, Path) or isinstance(source, str):
            if not Path(source).exists():
                raise FileNotFoundError(
                    f"Video source {source} does not exist."
                )

        self.cap = cv2.VideoCapture(source)

    def get_frame(self) -> tuple[bool, ndarray]:
        """Capture a single frame from the video source."""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        """Release the video capture object."""
        self.cap.release()
