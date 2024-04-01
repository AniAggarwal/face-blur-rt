import numpy as np
from numpy import ndarray


def rescale_boxes(boxes: ndarray, original_size: tuple[int, int]):
    """
    Rescale bounding box coordinates from fractional representation to absolute pixel values.
    Will skip processing if the input boxes are empty.

    Args:
        boxes: np.array of shape (n, 4) with each row being (x1, y1, x2, y2) as fractions of frame.
        original_size: tuple (width, height) of the original frame size in pixels.

    Returns:
    - scaled_boxes: np.array of shape (n, 4) with each row being (x1, y1, x2, y2) in absolute pixels.
    """
    if boxes.size == 0:
        return boxes

    # Extract the original frame dimensions
    width, height = original_size

    # Scale the coordinates
    scaled_boxes = boxes * np.array([width, height, width, height])

    return scaled_boxes.astype(int)
