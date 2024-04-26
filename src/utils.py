import numpy as np
from numpy import ndarray
import dlib


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


def dlib_rect_to_bbox(dlib_rect):
    """
    Converts a dlib.rectangle object into a bounding box tuple.

    Parameters:
    - dlib_rect: A dlib.rectangle object.

    Returns:
    - bbox: An array (x1, y1, x2, y2) representing the top-left and bottom-right corners of the rectangle.
    """
    x1 = dlib_rect.left()
    y1 = dlib_rect.top()
    x2 = dlib_rect.right()
    y2 = dlib_rect.bottom()
    return np.array([x1, y1, x2, y2])

def bbox_to_dlib_rect(bbox_array):
    """
    Converts an array or list containing bounding box coordinates [x1, y1, x2, y2]
    into a dlib.rectangle object.

    Parameters:
    - bbox_array: A list or array [x1, y1, x2, y2] defining the top-left and bottom-right corners.

    Returns:
    - A dlib.rectangle object defined by the provided coordinates.
    """
    if len(bbox_array) != 4:
        raise ValueError("Input array must have four elements [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox_array
    return dlib.rectangle(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))

def bbox_overlap(bbox1, bbox2):
    """
    Checks if the center point of bbox1 is within bbox2.

    Parameters:
    - bbox1: [x1, y1, x2, y2] where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.
    - bbox2: [x1, y1, x2, y2] where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.

    Returns:
    - True if the center of bbox1 is within bbox2, False otherwise.
    """
    # Calculate the center of bbox1
    center_x1 = (bbox1[0] + bbox1[2]) / 2
    center_y1 = (bbox1[1] + bbox1[3]) / 2

    # Check if the center of bbox1 is within bbox2
    return (bbox2[0] <= center_x1 <= bbox2[2]) and (bbox2[1] <= center_y1 <= bbox2[3])

