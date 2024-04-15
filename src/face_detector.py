from pathlib import Path
from abc import ABC, abstractmethod

from numpy import ndarray
import torch
import numpy as np
import cv2
from scrfd.scrfd import SCRFD
import utils

# redundant imports?
import onnx
import onnxruntime as ort


class FaceDetector(ABC):
    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int] = (640, 480)
    ) -> None:
        self.model_path = model_path
        self.det_res = det_res

    @abstractmethod
    def detect_faces(self, frame: ndarray) -> ndarray:
        """Detect faces in a frame.

        Args:
            frame: a numpy array of shape (height, width, channels) representing the frame.

        Returns:
            A numpy array of shape (n, 4), where n is the number of detected faces.
            Each face instance contains four elements: x1, y1, x2, y2, where (x1, y1) and (x2, y2),
            with each pair representing the top-left and bottom-right corners of the bounding box, respectively.
            Note they are fractions of the frame's width and height, respectively.
        """
        # Implementation of face detection.
        return np.array([])  # Return list of detected faces' bounding boxes.


class YuNetDetector(FaceDetector):

    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int] = (320, 320)
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

    def detect_faces(self, frame: ndarray) -> ndarray:
        frame = cv2.resize(frame, self.det_res)
        num, faces = self.detector.detect(frame)
        if faces is None:
            return np.array([])

        # faces is 2D arr, rows = detected face instances, cols are:
        # x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
        # where x1, y1, w, h are the top-left coordinates, width and height of the face bounding box
        # {x, y}_{re, le, nt, rcm, lcm}: coordinates of right eye, left eye, nose tip, right corner, left corner of mouth

        # for now, discard everything but the face bounding box
        bboxes = faces[:, :4]
        # convert width and height to x2, y2
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        # now convert to fractions of frame size
        bboxes /= np.array([*self.det_res, *self.det_res])

        # make sure to clip to [0, 1]
        bboxes = np.clip(bboxes, 0, 1)

        return bboxes.astype(np.float32)

# TODO: get the bounding boxes working correctly
class ULFGLightDetector(FaceDetector):
    def __init__(
        self, model_path: str | Path, det_res: tuple[int, int] = (320, 240)
    ) -> None:
        super().__init__(model_path, det_res)

        detector = onnx.load(model_path)
        onnx.checker.check_model(detector)
        onnx.helper.printable_graph(detector.graph)
        #detector = backend.prepare(detector, device="CPU")  # default CPU

        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def detect_faces(self, frame: ndarray) -> ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.det_res)
        # save this for medium version
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        threshold = 0.7
        return self.predict(frame.shape[1], frame.shape[0], confidences, boxes, threshold)
    
    #currently not working
    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = utils.hard_nms(box_probs,
                               iou_threshold=iou_threshold,
                               top_k=top_k,
                               )
            picked_box_probs.append(box_probs)
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.float32)


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

