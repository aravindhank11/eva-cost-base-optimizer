# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import numpy as np
import pandas as pd

from eva.models.catalog.frame_info import FrameInfo
from eva.models.catalog.properties import ColorSpace
from eva.udfs.abstract.pytorch_abstract_udf import PytorchAbstractClassifierUDF
import sys
import cv2
import importlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from torch import Tensor
except ImportError as e:
    raise ImportError(
        f"Failed to import with error {e}, \
        please try `pip install torch`"
    )

try:
    import torchvision
except ImportError as e:
    raise ImportError(
        f"Failed to import with error {e}, \
        please try `pip install torch`"
    )


class ObjectDetector(PytorchAbstractClassifierUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    def __init__(self, model_name):
        module = importlib.import_module("torchvision.models.detection")
        self._torch_model = getattr(module, model_name)
        self.model_name = model_name
        super().__init__()

    @property
    def name(self) -> str:
        return self.model_name

    def setup(self, threshold=0.85):
        self.threshold = threshold
        self.model = self._torch_model(pretrained=True, progress=False)
        self.model.eval()

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return [
            "__background__",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "N/A",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "N/A",
            "backpack",
            "umbrella",
            "N/A",
            "N/A",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "N/A",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "N/A",
            "dining table",
            "N/A",
            "N/A",
            "toilet",
            "N/A",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "N/A",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def forward(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])

        """
        predictions = self.model(frames)
        outcome = pd.DataFrame()
        for prediction in predictions:
            pred_class = [
                str(self.labels[i]) for i in list(self.as_numpy(prediction["labels"]))
            ]
            pred_boxes = [
                [i[0], i[1], i[2], i[3]]
                for i in list(self.as_numpy(prediction["boxes"]))
            ]
            pred_score = list(self.as_numpy(prediction["scores"]))
            valid_pred = [pred_score.index(x) for x in pred_score if x > self.threshold]

            if valid_pred:
                pred_t = valid_pred[-1]
            else:
                pred_t = -1

            pred_boxes = np.array(pred_boxes[: pred_t + 1])
            pred_class = np.array(pred_class[: pred_t + 1])
            pred_score = np.array(pred_score[: pred_t + 1])
            outcome = outcome.append(
                {"labels": pred_class, "scores": pred_score, "bboxes": pred_boxes},
                ignore_index=True,
            )
        return outcome

class Resnet50ObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__("fasterrcnn_resnet50_fpn_v2")

class MobilenetObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__("fasterrcnn_mobilenet_v3_large_fpn")

class SsdLiteObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__("ssdlite320_mobilenet_v3_large")

class SsdVggObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__("ssd300_vgg16")

class KeypointObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__("keypointrcnn_resnet50_fpn")

if __name__ == "__main__":
    # Read the image
    frame_path = sys.argv[1]
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)


    # Test Yolov5x6
    for class_name in [KeypointObjectDetector, Resnet50ObjectDetector, MobilenetObjectDetector,
                       SsdLiteObjectDetector, SsdVggObjectDetector]:
        obj = class_name()
        tensor_frame = obj.transform(img)
        out = obj.forward(tensor_frame)
        print(str(class_name), out)
