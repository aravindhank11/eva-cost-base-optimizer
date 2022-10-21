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


class UdfProfileMetadata(BaseModel):
    __tablename__ = "udfprofile"

    _batch_size = Column("batch_size", Integer)
    _time_taken = Column("time_taken", Integer)
    _accuracy = Column("accuracy", Float)
    _udf_id = Column("udf_id", Integer, ForeignKey("udf.id"))

    def __init__(self, udf_id: int, batch_size: int, time_taken: int, accuracy: float):
        self._udf_id = udf_id
        self._batch_size = batch_size
        self._time_taken = time_taken
        self._accuracy = accuracy

    @property
    def id(self):
        return self._id

    @property
    def udf_id(self):
        return self._udf_id

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def time_taken(self):
        return self._time_taken

    @property
    def accuracy(self):
        return self._accuracy

    def __str__(self):
        udf_profile_str = "udf_profile: ({}, {}, {}, {} {})\n".format(
            self.id, self.udf_id, self.batch_size, self.time_taken, self.accuracy
        )
        return udf_str

    def __eq__(self, other):
        return (
            self.id == other.id
            and self.udf_id == other.udf_id
            and self.batch_size == other.batch_size
            and self.time_taken == other.time_taken
            and self.accuracy == other.accuracy
        )

    def __hash__(self) -> int:
        return hash((self.id, self.udf_id, self.batch_size, self.time_taken, self.accuracy))
