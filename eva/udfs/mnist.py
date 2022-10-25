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

from eva.models.catalog.frame_info import FrameInfo
from eva.models.catalog.properties import ColorSpace
from eva.udfs.abstract.pytorch_abstract_udf import PytorchAbstractClassifierUDF

from typing import List
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize
import os, sys
import pandas as pd
from torch import Tensor, nn
import cv2

class CNN(nn.Module):
    def __init__(self, num_classes, conv_dims, fc_dims):
        super().__init__()
        assert len(conv_dims) > 0, 'conv_dims can not be empty'
        assert len(fc_dims) > 0, 'fc_dims can not be empty'

        convs, fcs = [], []
        for i in range(len(conv_dims)):
            in_dims = 1 if i == 0 else conv_dims[i - 1]
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_dims, conv_dims[i], 5),
                    nn.BatchNorm2d(conv_dims[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2)
                )
            )

        for i in range(len(fc_dims) - 1):
            fcs.append(
                nn.Sequential(
                    nn.Linear(fc_dims[i], fc_dims[i + 1]),
                    nn.BatchNorm1d(fc_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        fcs.append(nn.Linear(fc_dims[-1], num_classes))

        self.conv = nn.Sequential(*convs)
        self.fc = nn.Sequential(*fcs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class RNN(nn.Module):
    def __init__(self, arch, in_dim, num_classes, hidden_size=64, num_layers=1):
        super().__init__()
        assert arch in ['RNN', 'LSTM', 'GRU'], 'Unrecognized model name'
        if arch == 'RNN':
            net = nn.RNN
        elif arch == 'LSTM':
            net = nn.LSTM
        else:
            net = nn.GRU
        self.rnn = net(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: [B, L, C]
        :return:
        """
        x = x.squeeze(1)
        r_out, _ = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out = self.softmax(out)

        return out

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims):
        super().__init__()
        assert len(hidden_dims) > 0, 'hidden_dims can not be empty'

        fcs = []
        for i in range(len(hidden_dims)):
            in_dim = in_dim if i == 0 else hidden_dims[i - 1]
            fcs.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(inplace=True)
                )
            )
        fcs.append(nn.Linear(hidden_dims[-1], num_classes))

        self.fc = nn.Sequential(*fcs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

class Mnist(PytorchAbstractClassifierUDF, nn.Module):
    def __init__(self, model_name):
        self.__model_name = model_name
        self.__model_path = "model"

        use_gpu = torch.cuda.is_available()
        self.__device = torch.device("0" if use_gpu else "cpu")

        self.__config = {
            "CNN" : {
                "num_classes": 10,
                "conv_dims"  : [6, 16],
                "fc_dims"    : [256, 120, 84],
            },
            "RNN" : {
                "in_dim"      : 28,
                "num_classes" : 10,
                "hidden_size" : 64,
                "num_layers"  : 1,
            },
            "MLP" : {
                "in_dim"      : 784,
                "num_classes" : 10,
                "hidden_dims" : [256, 120, 84],
            },
        }

        # Call super's init
        PytorchAbstractClassifierUDF.__init__(self)

    @property
    def name(self) -> str:
        return "CnnMnist"

    @property
    def input_format(self):
        return FrameInfo(1, 28, 28, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return list([str(num) for num in range(10)])

    def __build_model(self):
        assert self.__model_name in ['RNN', 'LSTM', 'GRU', 'MLP', 'CNN'], 'Unrecognized model name'

        if self.__model_name == "MLP":
            model = MLP(
                in_dim      = self.__config["MLP"]["in_dim"],
                num_classes = self.__config["MLP"]["num_classes"],
                hidden_dims = self.__config["MLP"]["hidden_dims"]
            )
        elif self.__model_name == "CNN":
            model = CNN(
                num_classes  = self.__config["CNN"]["num_classes"],
                conv_dims    = self.__config["CNN"]["conv_dims"],
                fc_dims      = self.__config["CNN"]["fc_dims"]
            )
        else:
            model = RNN(
                arch        = self.__model_name,
                in_dim      = self.__config["RNN"]["in_dim"],
                num_classes = self.__config["RNN"]["num_classes"],
                hidden_size = self.__config["RNN"]["hidden_size"],
                num_layers  = self.__config["RNN"]["num_layers"]
            )

        return model

    def setup(self):
        print("In setup")
        # Load model
        self.__model = self.__build_model()

        # Move to device
        self.__model.to(self.__device)

        # Load State
        model_filename = os.path.join(self.__model_path, self.__model_name + '.pkl')
        state_dict = torch.load(model_filename, map_location=self.__device)
        self.__model.load_state_dict(state_dict)

        # Put in eval mode
        self.__model.eval()

    def forward(self, frames: Tensor) -> pd.DataFrame:
        images = frames.to(self.__device)
        outputs = self.__model(images)
        preds = torch.max(outputs, dim=-1)[1]
        preds_pd = pd.DataFrame(preds.numpy())
        preds_pd.rename(columns = {0: "label"}, inplace=True)
        return preds_pd

    def transform(self, images) -> Compose:
        composed = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        # reverse the channels from opencv
        return composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)

    def get_device(self):
        return self.__device

class CNNMnist(Mnist):
    def __init__(self):
        super().__init__("CNN")

class RNNMnist(Mnist):
    def __init__(self):
        super().__init__("RNN")

class MLPMnist(Mnist):
    def __init__(self):
        super().__init__("MLP")

class LSTMMnist(Mnist):
    def __init__(self):
        super().__init__("LSTM")

class GRUMnist(Mnist):
    def __init__(self):
        super().__init__("GRU")

if __name__ == "__main__":
    # Read the image
    frame_path = sys.argv[1]
    img = cv2.imread(frame_path, 0)

    # Test CNN
    obj = CNNMnist()
    obj.setup()
    tensor_frame = obj.transform(img)
    out = obj.forward(tensor_frame)
    print("CNN", out)

    # Test RNN
    obj = RNNMnist()
    obj.setup()
    tensor_frame = obj.transform(img)
    out = obj.forward(tensor_frame)
    print("RNN", out)

    # Test LSTM
    obj = LSTMMnist()
    obj.setup()
    tensor_frame = obj.transform(img)
    out = obj.forward(tensor_frame)
    print("LSTM", out)

    # Test MLP
    obj = MLPMnist()
    obj.setup()
    tensor_frame = obj.transform(img)
    out = obj.forward(tensor_frame)
    print("MLP", out)

    # Test GRU
    obj = GRUMnist()
    obj.setup()
    tensor_frame = obj.transform(img)
    out = obj.forward(tensor_frame)
    print("GRU", out)
