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

import numpy as np


def scale_bboxes(bboxes, scale=1.2):
    """
    Upscales an array of bounding boxes by a given scale around their centers.

    Args:
    bboxes (np.array): A numpy array with shape (n, 4) where each row is a bounding box in the format [xmin, ymin, xmax, ymax].
    scale (float): The scaling factor, default is 1.2 for 20% increase.

    Returns:
    np.array: A new numpy array containing the upscaled bounding boxes.
    """

    if bboxes.size == 0:
        return bboxes

    # Initialize the output array
    upscaled_bboxes = np.zeros_like(bboxes)
    
    # Process each bounding box
    for i, bbox in enumerate(bboxes):
        # Extract the corners
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate the center of the bounding box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        # Calculate new half widths/heights
        half_width = (xmax - xmin) * scale / 2
        half_height = (ymax - ymin) * scale / 2
        
        # Define the new corners based on the new size and same center
        new_xmin = center_x - half_width
        new_ymin = center_y - half_height
        new_xmax = center_x + half_width
        new_ymax = center_y + half_height
        
        # Store the new bounding box
        upscaled_bboxes[i] = [new_xmin, new_ymin, new_xmax, new_ymax]

    return upscaled_bboxes