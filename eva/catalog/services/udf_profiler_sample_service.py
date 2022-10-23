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
from sqlalchemy.orm.exc import NoResultFound

from eva.catalog.models.udf_profiler_sample import UdfProfilerSample
from eva.catalog.services.base_service import BaseService
from eva.utils.logging_manager import logger
from eva.utils.metrics import Metrics


class UdfProfilerSampleService(BaseService):
    def __init__(self):
        super().__init__(UdfProfilerSample)

    def create_udf_profiler_sample(self, udf_type: str, sample_path: str, validation_path: str) -> UdfProfilerSample:
        """Creates a new udf profiler sample for a type with given paths

        Arguments:
             udf_type (str): type of the UDF. e.g.: ObjectDetector
             sample_path (str): path to sample video file
             validation_path (str): path to sample validation CSV file

        Returns:
            UdfProfilerSample: Returns the new entry created
        """
        metadata = self.model(udf_type,
                              sample_path,
                              validation_path)
        metadata = metadata.save()
        self.print_all_profiler_samples("Post Inserting {}".format(udf_type))
        return metadata
    
    def get_udf_profiler_sample_by_type(self, udf_type: str):
        """
        Gets UDF profiler sample info by given type
        """
        try:
            result = self.model.query.filter(
                self.model._udf_type == udf_type
            ).all()
            return result
        except Exception as e:
            error = f"Getting profiler sample for {udf_type} raised {e}"
            logger.error(error)
            raise RuntimeError(error)

    def drop_udf_profiler_sample(self, udf_type: str):
        """Drop a udf profiler sample entry from the catalog udfprofilersample

        Arguments:
            udf_type (str): udf type to be deleted

        Returns:
            True if successfully deleted else False
        """
        return_val = True
        try:
            list_of_udf_profiler_samples = self.model.query.filter(self.model._udf_type == udf_type)
            for udf_profiler_sample in list_of_udf_profiler_samples:
                udf_profiler_sample.delete()
        except Exception:
            logger.exception("Delete udf Profile failed for id={}".format(name))
            return_val = False

        self.print_all_profiler_samples("Post Dropping {}".format(udf_type))
        return return_val

    def print_all_profiler_samples(self, when=""):
        print(when)
        list_of_udf_profiler_samples = self.get_all_profiler_samples()
        for udf_profiler_sample in list_of_udf_profiler_samples:
            print("  {} {} {} {}".format(
                udf_profiler_sample._id,
                udf_profiler_sample._udf_type,
                udf_profiler_sample._sample_path,
                udf_profiler_sample._validation_path,
            ))

    def get_all_profiler_samples(self):
        try:
            return self.model.query.all()
        except NoResultFound:
            return []
