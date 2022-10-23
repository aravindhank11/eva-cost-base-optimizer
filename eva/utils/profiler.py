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
        print("testing")
        metrics_list = []
        vidcap = cv2.VideoCapture("mnist.mp4")
        _, image = vidcap.read()
        batch_sizes = [1, 10, 100, 1000]
        for batch in batch_sizes:
            # metrics_obj = Metrics()
            frame_arr = np.zeros(shape=(batch, 28, 28, 3))
            for i in range(batch):
                frame_arr[i] = image
                _, image = vidcap.read()
            batched_tensor = torch.tensor(frame_arr)
            start_time = time.time()
            self._classobj.forward(batched_tensor)
            time_taken = time.time() - start_time
            batch_size = batch
            # TO DO
            accuracy = 100
            metrics_obj = Metrics(time_taken, accuracy, batch_size)
            metrics_list.append(metrics_obj)

        return metrics_list
