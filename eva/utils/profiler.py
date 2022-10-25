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
from sqlalchemy import true
import torch
import pandas as pd
import importlib.util

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
        frame_list = []
        ret = 1
        while ret:            
            ret, img = vidcap.read()
            frame_list.append(img)
        print("tot ex frames: {}".format(len(frame_list)))
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
        # print(labels_ip)
        print("label shape: {}".format(type(labels_ip)))

        
        batch_sizes = []
        iterator = 1
        while iterator <= total_frames:
            batch_sizes.append(iterator)
            iterator *= 5
        print("Batch Sizes: {}".format(batch_sizes))
        for id, batch in enumerate(batch_sizes): # 1 5 25 125
            frame_arr = np.zeros(shape=(batch, c, w, h))
            print("for batch {}".format(batch))
            accuracy_avg = 0.0
            time_avg = 0.0
            batch_size = batch
            iter = batch
            iter_prev = 0
            sum_acc = 0.0
            sum_time = 0.0
            counter = 0
            while(iter < total_frames):
                # in while loop use: iter_prev <= total_frames
                # if iter > total_frames:
                #     continue
                    # iter = total_frames
                    # print("last case: {}".format(iter))
                label_each_round = []
                print("round start: {} {}".format(iter_prev, iter))
                for i in range(iter_prev, iter):                    
                    image = frame_list[i]
                    label_each_round.append(labels_ip[i])
                    if(c==1):
                        #grayscale
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = gray.reshape(c,w,h)
                    else:
                        image = image.reshape(c,w,h)
                    frame_arr[i%batch] = image
                # print("len of batch: {}".format(len(frame_arr)))
                batched_tensor = torch.tensor(frame_arr).float()
                # print(batched_tensor.shape)
                start_time = time.time()
                predictions = self._classobj.forward(batched_tensor)
                label_each_round = pd.Series(label_each_round)
                print("pred shape: {} grnd truth shape: {}".format(predictions.label.shape, label_each_round.shape))
                time_taken = time.time() - start_time
                spec = importlib.util.spec_from_file_location("UdfAccuracy", "eva/utils/accuracy_impl.py")
                accuracy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(accuracy_module)
                accuracy = accuracy_module.UdfAccuracy.calculate_accuracy(label_each_round, predictions.label)
                print("mil gaya {}".format(accuracy))
                sum_time += time_taken
                sum_acc += accuracy
                iter_prev = iter
                iter += batch
                counter += 1
                # print("round over: {} {}".format(iter_prev, iter))
            accuracy_avg = sum_acc / counter
            time_avg = sum_time / counter
            print("acc: {} time: {} rounds: {}".format(accuracy_avg, time_avg, counter))

            # TODO: Calculate accuracy from python file provided in init.
            
            metrics_obj = Metrics(batch_size, time_avg, accuracy_avg)
            metrics_list.append(metrics_obj)
        return metrics_list
