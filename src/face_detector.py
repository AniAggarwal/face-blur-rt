from pathlib import Path
from abc import ABC, abstractmethod

from numpy import ndarray
import torch
import numpy as np
import cv2
from scrfd.scrfd import SCRFD


class FaceDetector(ABC):
    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int] = (640, 480)
    ) -> None:
        self.model_path = model_path
        self.det_res = det_res

    @abstractmethod
    def detect_faces(self, frame: ndarray) -> tuple[ndarray, ndarray]:
        """Detect faces in a frame.

        Args:
            frame: a numpy array of shape (height, width, channels) representing the frame.

        Returns:
            A numpy array of shape (n, 4), where n is the number of detected faces and the full faces detction.
            Each face instance contains four elements: x1, y1, x2, y2, where (x1, y1) and (x2, y2),
            with each pair representing the top-left and bottom-right corners of the bounding box, respectively.
            Note they are fractions of the frame's width and height, respectively.
            The full faces detection is the output of the face detector model and may have more info.
        """
        # Implementation of face detection.
        return np.array([])  # Return list of detected faces' bounding boxes.


class YuNetDetector(FaceDetector):

    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int] = (640, 480)
    ) -> None:
        super().__init__(model_path, det_res)
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            self.det_res,
            0.5,  # confidence threshold
            0.5,  # nms threshold
            10,  # top_k
        )
        self.detector.setInputSize(self.det_res)

    def detect_faces(self, frame: ndarray) -> tuple[ndarray, ndarray]:
        frame = cv2.resize(frame, self.det_res)
        _, features = self.detector.detect(frame)
        if features is None:
            return np.array([]), np.array([])

        # faces is 2D arr, rows = detected face instances, cols are:
        # x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
        # where x1, y1, w, h are the top-left coordinates, width and height of the face bounding box
        # {x, y}_{re, le, nt, rcm, lcm}: coordinates of right eye, left eye, nose tip, right corner, left corner of mouth

        # for now, discard everything but the face bounding box
        bboxes = features[:, :4]

        # convert width and height to x2, y2
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        # now convert to fractions of frame size
        bboxes /= np.array([*self.det_res, *self.det_res])

        # make sure to clip to [0, 1]
        bboxes = np.clip(bboxes, 0, 1)

        return bboxes.astype(np.float32), features


class SCRFDDetector(FaceDetector):
    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int]
    ) -> None:
        super().__init__(model_path, det_res)

        # for now try using pytorch native model
        self.model = torch.load(model_path)

        # TODO: figure this out
        # SCRFD requires onnx runtime; convert to onnx if not already
        # model_path = Path(model_path)
        # if not model_path.suffix == ".onnx":
        #     # Convert model to onnx
        #     pth = Path(model_path)
        #     onnx_model_path = pth.parent / (pth.stem + ".onnx")
        #     model = torch.load(model_path)
        #     dummy_input = torch.randn(3, *self.input_res)
        #     torch.onnx.export(
        #         model,
        #         dummy_input,
        #         str(onnx_model_path),
        #     )
        # else:
        #     onnx_model_path = model_path
        #
        # self.model = SCRFD(self.model_path)
        # self.model.prepare(-1)

    def detect_faces(self, frame: ndarray) -> list[int]:
        # pytorch implementation
        print(self.model)
        print(self.model(frame))

        return []

    # TODO: make onnx implementation work
    # def detect_faces(self, frame: ndarray) -> list[int]:
    #     # Implementation of face detection using SCRFD.
    #
    #     ta = datetime.datetime.now()
    #
    #     bboxes, kpss = self.model.detect(frame, 0.5, input_size=self.input_res)
    #
    #     tb = datetime.datetime.now()
    #     print("all cost:", (tb - ta).total_seconds() * 1000)
    #
    #     if kpss is not None:
    #         print("kpss:", kpss.shape)
    #
    #     print("bboxes shape:", bboxes.shape)
    #
    #     for i in bboxes:
    #         x1, y1, x2, y2, score = bboxes[i].astype(np.int32)
    #         print("bbox score:", score)
    #         bboxes[i] = [x1, y1, x2, y2]
    #
    #     # Return list of detected faces' bounding boxes.
    #     return bboxes
