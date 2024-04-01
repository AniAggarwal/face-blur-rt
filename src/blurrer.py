from numpy import ndarray


class Blurrer:
    def __init__(self, method: str = "bounding_box") -> None:
        self.method = method

    def apply_blur(self, frame: ndarray, faces: list) -> ndarray:
        """Apply the specified blurring technique to the faces in the frame."""
        # Implementation of blurring techniques.
        return frame
