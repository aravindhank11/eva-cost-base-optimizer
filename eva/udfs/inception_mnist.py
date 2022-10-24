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
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale, Resize, CenterCrop
from torchvision import models
import torchvision.models as models
import torch
import torch.nn as nn

from PIL import Image

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


class InceptionMnist(PytorchAbstractClassifierUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return "InceptionMnist"

    def setup(self):
        #self.threshold = threshold
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.in_features, 10)
        self.model.eval()

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(3, 28, 28, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return list([str(num) for num in range(10)])

    def transform(self, images) -> Compose:
        composed = Compose([
            Resize(312),
            CenterCrop(299),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # reverse the channels from opencv
        return composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)
        
    def forward(self, frames: Tensor) -> pd.DataFrame:
        outcome = pd.DataFrame()
        predictions = self.model(frames)
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            outcome = outcome.append({"label" : str(label)}, ignore_index=True)
        
        return outcome
