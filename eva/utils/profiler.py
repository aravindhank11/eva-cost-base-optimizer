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
import os.path
import sys

from eva.utils.metrics import Metrics


class Profiler:
    def __init__(self, filepath: str, classname: str, samplepath: str, validationpath: str):
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

        self._samplepath = samplepath
        self._validationpath = validationpath

    def run(self):
        """
        * Profiles the given UDF for various batch sizes
        * Gathers metrics related to time_taken

        Param: None

        Returns: List[Metrics]
        """        

        #NOTE This wont work if input_format is -1. Need to implement for that case.

        expected_h = self._classobj.input_format.height
        expected_w = self._classobj.input_format.width
        expected_c = self._classobj.input_format.channels

        batch_sizes = [5, 20, 50, 200, 400, 500, 750] 
        metrics_list=[]
        for batch in enumerate(batch_sizes):
            input_tensor = torch.rand(batch, expected_c, expected_w, expected_h)
            start_time = time.time()
            predictions = self._classobj.forward(input_tensor)
            time_taken = time.time() - start_time
            metrics_obj = Metrics(batch, time_taken)
            metrics_list.append(metrics_obj)
        return metrics_list            

