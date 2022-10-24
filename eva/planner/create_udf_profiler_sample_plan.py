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
from pathlib import Path
from typing import List

from eva.catalog.models.udf_io import UdfIO
from eva.planner.abstract_plan import AbstractPlan
from eva.planner.types import PlanOprType


class CreateUDFProfilerSamplePlan(AbstractPlan):
    """
    This plan is used for storing information required to create udf operators

    Attributes:
        if_not_exists: bool
            if true should throw an error if udf with same name exists
            else will replace the existing
        udf_type: str
            udf type. it ca be object detection, classification etc.
        sample_path: str
            file path which holds the sample for the udf_type.
            This file should be placed in the UDF directory and
            the path provided should be relative to the UDF dir.
        validation_path: str
            file path which holds the validation data for the sample at sample_path.
            This file should be placed in the UDF directory and
            the path provided should be relative to the UDF dir.
    """

    def __init__(
        self,
        if_not_exists: bool,
        udf_type: str = None,
        sample_path: Path,
        validation_path: Path
    ):
        super().__init__(PlanOprType.CREATE_UDF_PROFILER_SAMPLE)
        self._if_not_exists = if_not_exists
        self._udf_type = udf_type
        self._sample_path = sample_path
        self._validation_path = validation_path

    @property
    def if_not_exists(self):
        return self._if_not_exists

    @property
    def udf_type(self):
        return self._udf_type

    @property
    def sample_path(self):
        return self._sample_path

    @property
    def validation_path(self):
        return self._validation_path

    def __hash__(self) -> int:
        return hash(
            (
                super().__hash__(),
                self.if_not_exists,
                self.udf_type,
                self.sample_path,
                self.validation_path
            )
        )
