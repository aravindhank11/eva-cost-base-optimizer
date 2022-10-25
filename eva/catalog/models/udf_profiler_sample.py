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
from sqlalchemy import Column, String, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship

from eva.catalog.models.base_model import BaseModel


class UdfProfilerSample(BaseModel):
    __tablename__ = "udfprofilersample"

    _udf_type = Column("udf_type",String)
    _sample_path = Column("sample_path",String)
    _validation_path = Column("validation_path",String)

    def __init__(self, udf_type: str, sample_path: str, validation_path: str):
        self._udf_type = udf_type
        self._sample_path = sample_path
        self._validation_path = validation_path

    @property
    def id(self):
        return self._id

    @property
    def udf_type(self):
        return self._udf_type

    @property
    def sample_path(self):
        return self._sample_path
    
    @property
    def validation_path(self):
        return self._validation_path

    def __str__(self):
        udf_profiler_sample_str = "udf_profiler_sample: ({}, {}, {}, {})\n".format(
            self.id, self.udf_type, self.sample_path, self.validation_path
        )
        return udf_str

    def __eq__(self, other):
        return (
            self.id == other.id
            and self.udf_type == other.udf_type
            and self.sample_path == other.sample_path
            and self.validation_path == other.validation_path
        )

    def __hash__(self) -> int:
        return hash((self.id, self.udf_type, self.sample_path, self.validation_path))
