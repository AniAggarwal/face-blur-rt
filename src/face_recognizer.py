from pathlib import Path
from numpy import ndarray
import cv2

import utils
from face_detector import FaceDetector


class FaceRecognizer:
    def __init__(
        self, model_path: str | Path, known_faces_path: str | Path
    ) -> None:
        self.model_path = model_path
        self.known_faces_path = known_faces_path

    def recognize_faces(
        self, frame: ndarray, faces: ndarray
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
    ):
        super().__init__(model_path, known_faces_path)

        self.recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")

        # create a dictionary of known faces, mapping name of directory to list of face encodings
        self.face_encodings = {}

        # iterate over each image within the directory, algin it, and save its features
        for name in Path(self.known_faces_path).iterdir():
            self.face_encodings[name.name] = []
            for image_path in name.iterdir():
                image = cv2.imread(str(image_path))
                # to align, we must first detect the face
                bboxes = detector_obj.detect_faces(image)
                bboxes = utils.rescale_boxes(bboxes, image.shape[:2])
                self.face_encodings[name.name].append(
                    self.recognizer.feature(
                        self.recognizer.alignCrop(image, bboxes)
                    )
                )

    def recognize_faces(
        self, frame: ndarray, faces: ndarray
    ) -> dict[int, str | None]:
        COSINE_THRESH = 0.363
        # iterate over detected faces and process them, check against
        # each known face, and map the index to the name if possible
        output = {}
        for i, face in enumerate(faces):
            # default recognized face to None
            output[i] = None

            bboxes = utils.rescale_boxes(face, frame.shape[:2])
            curr = self.recognizer.feature(
                self.recognizer.alignCrop(frame, bboxes)
            )

            for name, encodings in self.face_encodings.items():
                for encoding in encodings:
                    cosine_score = self.recognizer.match(
                        encoding, curr, cv2.FaceRecognizerSF_FR_COSINE
                    )
                    print("cosine score", cosine_score)

                    if cosine_score >= COSINE_THRESH:
                        output[i] = name
                        # TODO: add count threshold where enough matches are needed
                        break

                # only allow one recognized face per detected face
                # TODO: add tie breaking mechanism; currently chooses first one
                if output[i] is not None:
                    break

        return output
