from enum import Enum, auto
from numpy import ndarray
import cv2
import numpy as np


class BlurringMethod(Enum):
    NONE = auto()
    LINE = auto()
    BLACK = auto()
    GAUSSIAN = auto()
    PIXELATE = auto()
    MOSAIC = auto()


class BlurringShape(Enum):
    NONE = auto()
    SQUARE = auto()
    CIRCLE = auto()


class Blurrer:
    def __init__(self, method: BlurringMethod, shape: BlurringShape) -> None:
        self.method = method
        self.shape = shape

    def apply_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
        """Apply the specified blurring technique to the faces in the frame."""
        if (
            self.shape == BlurringShape.NONE
            or self.method == BlurringMethod.NONE
        ):
            return frame

        if self.shape == BlurringShape.CIRCLE:
            faces = self.get_circle_center(faces)

        if self.method == BlurringMethod.LINE:
            frame = self.line_blur(frame, faces)
        elif self.method == BlurringMethod.BLACK:
            frame = self.black_blur(frame, faces)
        elif self.method == BlurringMethod.GAUSSIAN:
            frame = self.gaussian_blur(frame, faces)

        # elif self.method == BlurringMethod.PIXELATE:
        #     frame = self.pixelate_blur(frame, faces)
        # elif self.method == BlurringMethod.MOSAIC:
        #     frame = self.mosaic_blur(frame, faces)

        return frame

    def get_circle_center(self, bboxes: ndarray) -> ndarray:
        output = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            radius = (x2 - x1) // 2
            output.append([*center, radius])

        return np.array(output)

    def line_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
        for face in faces:
            if self.shape == BlurringShape.SQUARE:
                x1, y1, x2, y2 = face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif self.shape == BlurringShape.CIRCLE:
                x, y, radius = face
                cv2.circle(frame, (x, y), radius, (255, 0, 0), 2)

        return frame

    def black_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
        """Apply a black box blur to the faces in the frame."""
        for face in faces:
            if self.shape == BlurringShape.SQUARE:
                x1, y1, x2, y2 = face
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), -1) 

            elif self.shape == BlurringShape.CIRCLE:
                x, y, radius = face
                cv2.circle(frame, (x, y), radius, (0, 0, 0), -1)

        return frame

    def gaussian_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
        """Apply a Gaussian blur to the faces in the frame."""
        height, width = frame.shape[:2]
        for face in faces:
            if self.shape == BlurringShape.SQUARE:
                x1, y1, x2, y2 = face
                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))
                

                print(face)
                face_region = frame[y1:y2, x1:x2]
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y2, x1:x2] = face_region
            elif self.shape == BlurringShape.CIRCLE:
                x, y, radius = face
                center = (x, y)
                # we will make a circular mask, grab that portion of the image, apply a blur
                # and finally paste it back
                mask = np.zeros_like(frame)
                cv2.circle(mask, center, radius, (255, 255, 255), -1)
                frame_blurred = cv2.GaussianBlur(frame, (99, 99), 30)
                frame = np.where(mask > 0, frame_blurred, frame)

        return frame

    # TODO: implement these
    # def pixelate_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
    #     """Apply a pixelation blur to the faces in the frame."""
    #     for face in faces:
    #         x1, y1, x2, y2 = face
    #         face_region = frame[y1:y2, x1:x2]
    #         face_region = cv2.resize(
    #             face_region, (30, 30), interpolation=cv2.INTER_LINEAR
    #         )
    #         face_region = cv2.resize(
    #             face_region,
    #             (x2 - x1, y2 - y1),
    #             interpolation=cv2.INTER_NEAREST,
    #         )
    #         frame[y1:y2, x1:x2] = face_region
    #     return frame
    #
    # def mosaic_blur(self, frame: ndarray, faces: ndarray) -> ndarray:
    #     """Apply a mosaic blur to the faces in the frame."""
    #     for face in faces:
    #         x1, y1, x2, y2 = face
    #         face_region = frame[y1:y2, x1:x2]
    #         face_region = cv2.resize(
    #             face_region, (10, 10), interpolation=cv2.INTER_NEAREST
    #         )
    #         face_region = cv2.resize(
    #             face_region,
    #             (x2 - x1, y2 - y1),
    #             interpolation=cv2.INTER_NEAREST,
    #         )
    #         frame[y1:y2, x1:x2] = face_region
    #     return frame
