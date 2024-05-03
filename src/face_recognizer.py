from pathlib import Path
import numpy as np
import cv2

import utils
from face_detector import FaceDetector
import pdb


class FaceRecognizer:
    def __init__(self, model_path: str | Path, known_faces_path: str | Path, threshold) -> None:
        self.model_path = model_path
        self.known_faces_path = known_faces_path
        self.threshold = threshold

    def recognize_faces(
        self, frame: np.ndarray, faces: np.ndarray
    ) -> dict[int, str | None]:
        """Identify specific faces within the frame."""
        # Implementation of face recognition.
        return {}  # Return list of recognized faces.


class SFRecognizer(FaceRecognizer):
    def __init__(
        self,
        model_path: str | Path,
        detector_obj: FaceDetector,
        known_faces_path: str | Path,
        threshold: float
    ):
        super().__init__(model_path, known_faces_path, threshold)

        self.recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")
        self.det_res = detector_obj.det_res
        # create a dictionary of known faces, mapping name of directory to list of face encodings
        self.face_encodings = {}
        self.threshold = threshold
        # iterate over each image within the directory, algin it, and save its features
        for name in Path(self.known_faces_path).iterdir():
            self.face_encodings[name.name] = []
            encoding = []
            for image_path in name.iterdir():
                image = cv2.imread(str(image_path))
                # to align, we must first detect the face
                image = cv2.resize(image, detector_obj.det_res)
                bboxes, recognition_faces = detector_obj.detect_faces(image)
                # self.face_encodings[name.name].append(
                #     self.recognizer.feature(
                #         self.recognizer.alignCrop(image, recognition_faces)
                #     )
                # )
                encoding.append(self.recognizer.feature(
                         self.recognizer.alignCrop(image, recognition_faces)
                     ))
            self.face_encodings[name.name] = np.mean(encoding, axis=0)

        print(
            f"Loaded {len(self.face_encodings)} known faces."
            f"Training images per face: { {n: len(f) for n, f in self.face_encodings.items()} }"
        )

    def recognize_faces(self, frame: np.ndarray, faces: np.ndarray) -> dict[int, str | None]:
        # iterate over detected faces and process them, check against
        # each known face, and map the index to the name if possible
        output = {}
        for i, face in enumerate(faces):
            # default recognized face to None
            output[i] = None
            # bboxes = utils.rescale_boxes(face, frame.shape[:2])
            curr = self.recognizer.feature(
                self.recognizer.alignCrop(cv2.resize(frame, self.det_res), face)
            )

            best_idx = None

            #Set a Cosine Threshold Here
            best_cosine = self.threshold

            for name, encoding in self.face_encodings.items():
                print("checking", name)

                # we will average the cosine similarity over each name
                avg_cosine = 0
                # for encoding in encodings:
                #     cosine_score = self.recognizer.match(
                #         encoding, curr, cv2.FaceRecognizerSF_FR_COSINE
                #     )
                #     print(f"comparing {name} with {cosine_score}")

                #     avg_cosine += cosine_score

                # avg_cosine /= len(encodings)
                avg_cosine = self.recognizer.match(
                        encoding, curr, cv2.FaceRecognizerSF_FR_COSINE
                    )
                print("avg cosine", avg_cosine)

                # select the best matching name
                if avg_cosine > best_cosine:
                    best_idx = name
                    best_cosine = avg_cosine

            # select the best match
            if best_idx is not None:
                output[i] = best_idx

        return output
