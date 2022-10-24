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
import pandas as pd

from eva.catalog.catalog_manager import CatalogManager
from eva.executor.abstract_executor import AbstractExecutor
from eva.models.storage.batch import Batch
from eva.planner.create_udf_profiler_sample_plan import CreateUDFProfilerSamplePlan
from eva.utils.generic_utils import path_to_class
from eva.utils.logging_manager import logger


class CreateUDFProfilerSampleExecutor(AbstractExecutor):
    def __init__(self, node: CreateUDFPlan):
        super().__init__(node)

    def validate(self):
        pass

    def exec(self):
        """Create udf executor

        Calls the catalog to create udf metadata.
        """
        catalog_manager = CatalogManager()

        # check catalog if it already has this udf entry
        # if catalog_manager.get_udf_by_name(self.node.name):
        #     if self.node.if_not_exists:
        #         msg = f"UDF {self.node.name} already exists, nothing added."
        #         logger.warn(msg)
        #         yield Batch(pd.DataFrame([msg]))
        #         return
        #     else:
        #         msg = f"UDF {self.node.name} already exists."
        #         logger.error(msg)
        #         raise RuntimeError(msg)
        # io_list = []
        # io_list.extend(self.node.inputs)
        # io_list.extend(self.node.outputs)
        sample_path = self.node.sample_path.absolute().as_posix()
        validation_path = self.node.validation_path.absolute().as_posix()

        #  TODO: Add logic to check TYPE in VALIDATION path py file here
        # check if we can create the udf object
        # try:
        #     path_to_class(impl_path, self.node.name)()
        # except Exception as e:
        #     err_msg = (
        #         f"{str(e)}. Please verify that the UDF class name in the "
        #         f"implementation file matches the provided UDF name {self.node.name}."
        #     )
        #     logger.error(err_msg)
        #     raise RuntimeError(err_msg)

        # create the actual udf
        udf_metadata = catalog_manager.create_udf_profiler_sample(
            self.node.udf_type, sample_path, validation_path
        )

        # Profile the UDF
        # sample = catalog_manager.get_udf_profiler_sample_by_type(self.node.udf_type)
        # profiler = Profiler(impl_path, self.node.name, sample.sample_path, sample.validation_path)
        # metrics = profiler.run()

        # Insert the profiled UDF to catalog
        # catalog_manager.create_udf_profile(udf_metadata.id, metrics)

        yield Batch(
            pd.DataFrame([f"UDF Profiler Sample {self.node.udf_type} successfully added to the database."])
        )
