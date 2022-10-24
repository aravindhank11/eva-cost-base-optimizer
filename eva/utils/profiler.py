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

import importlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd

from eva.utils.metrics import Metrics


class Profiler:
    def __init__(self, filepath: str, classname: str):
        """
        * Profiler is supposed to be called post basic validation
        * So object creation is guaranteed
        * Create an object of classname
        """
        abs_path = Path(filepath).resolve()
        spec = importlib.util.spec_from_file_location(abs_path.stem, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._classobj = getattr(module, classname)()

    def run(self):
        """
        * Profiles the given UDF for various batch sizes
        * Gathers metrics related to time_taken and accuracy

        Param: None

        Returns: List[Metrics]
        """
        # TODO: Implement the actual logic
        # Use self._classobj's methods to run for various batch sizes
        metrics_list = []
        # vidcap = cv2.VideoCapture("/home/azureuser/dbsi_project/eva-cost-base-optimizer/mnist_mini/mnist_mini.mp4")
        vidcap = cv2.VideoCapture('/home/naman/Desktop/eva-cost-base-optimizer/mnist_mini/mnist_mini.mp4')
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frames in the video: {}".format(total_frames))
        _, image = vidcap.read()
        h,w,c = image.shape
        
        expected_h = self._classobj.input_format.height
        expected_w = self._classobj.input_format.width
        expected_c = self._classobj.input_format.channels

        h = expected_h if expected_h != -1 else h
        w = expected_w if expected_w != -1 else w
        c = expected_c if expected_c != -1 else c

        if(c==1):
            #grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = gray.reshape(c,w,h)            
        else:
            image = image.reshape(c,w,h)
        
        print("image after reshape:{}".format(image.shape))

        # df = pd.read_csv('/home/azureuser/dbsi_project/eva-cost-base-optimizer/mnist_mini/mnist_mini_labels.csv')
        df = pd.read_csv('/home/naman/Desktop/eva-cost-base-optimizer/mnist_mini/mnist_mini_labels.csv', header=None)
        # print("last val check: {}".format(df.iloc[-1]))
        # print(df.iloc[-1])
        labels_ip = df.iloc[:,0]
        print(labels_ip)
        print("label shape: {}".format(labels_ip.shape))

        
        batch_sizes = []
        iterator = 1
        while iterator <= total_frames:
            batch_sizes.append(iterator)
            iterator *= 7
        for id, batch in enumerate(batch_sizes): # 1 5 25 125
            frame_arr = np.zeros(shape=(batch, c, w, h))
            # frame_arr = generate_tensor(batch_sizes, batch, frame_arr, c, w, h)
            if id != 0:
                for i in range(batch_sizes[id-1], batch):
                    frame_arr[i] = image
                    _, image = vidcap.read()
                    # print(image)
                    if(c==1):
                        #grayscale
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = gray.reshape(c,w,h)
                    else:
                        image = image.reshape(c,w,h)
            else:
                for i in range(batch):
                    frame_arr[i] = image
                    _, image = vidcap.read()
                    # print(image)
                    if(c==1):
                        #grayscale
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = gray.reshape(c,w,h)
                    else:
                        image = image.reshape(c,w,h)
            print("len of batch: {}".format(len(frame_arr)))
            batched_tensor = torch.tensor(frame_arr).float()
            print(batched_tensor.shape)
            start_time = time.time()
            data = self._classobj.forward(batched_tensor)
            # print("received data shape: {}".format(data.label.shape))
            time_taken = time.time() - start_time
            batch_size = batch
            # TO DO
            num_correct_pred = 0
            for id, label in enumerate(data.label):
                # print("{} {}".format(label, labels_ip[id]))
                if int(label) == (labels_ip[id]):
                    num_correct_pred += 1 
            # print("num_correct_pred: {} Total Predictions: {}".format(num_correct_pred, data.size))
            try:
                accuracy = (num_correct_pred/data.size) * 100
            except ZeroDivisionError:
                accuracy = -1
            metrics_obj = Metrics(time_taken, accuracy, batch_size)
            metrics_list.append(metrics_obj)
        return metrics_list


# def generate_tensor(batch_sizes, batch, frame_arr, c, w, h):
#     if id != 0:
#         for i in range(batch_sizes[id-1], batch):
#             frame_arr[i] = image
#             _, image = vidcap.read()
#             print(image)
#             if(c==1):
#                 #grayscale
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 image = gray.reshape(c,w,h)
#             else:
#                 image = image.reshape(c,w,h)
#     else:
#         for i in range(batch):
#             frame_arr[i] = image
#             _, image = vidcap.read()
#             print(image)
#             if(c==1):
#                 #grayscale
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 image = gray.reshape(c,w,h)
#             else:
#                 image = image.reshape(c,w,h)
#     return frame_arr