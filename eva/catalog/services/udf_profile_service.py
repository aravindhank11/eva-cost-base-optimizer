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
from eva.utils.generic_utils import path_to_class

from eva.catalog.models.udf_profile import UdfProfileMetadata
from eva.catalog.services.base_service import BaseService
from eva.utils.logging_manager import logger
from eva.utils.metrics import Metrics


class UdfProfileService(BaseService):
    def __init__(self):
        super().__init__(UdfProfileMetadata)

    def create_udf_profile(self, udf_id: int, metrics: Metrics) -> UdfProfileMetadata:
        """Creates a new udf profile entry
        Arguments:
            udf_id (int): udf_id corresponding to the metrics
            metrics: Metrics corresponding to the udf
        Returns:
            UdfMetadata: Returns the new entry created
        """
        metadata = self.model(udf_id,
                              metrics.batch_size,
                              metrics.time_taken)
        metadata = metadata.save()
        self.print_all_profile("Post Inserting {}".format(udf_id))
        return metadata

    def drop_udf_profile(self, udf_id: int):
        """Drop a udf profile entry from the catalog udfprofilemetadata
        Arguments:
            udf_id (int): udf id to be deleted
        Returns:
            True if successfully deleted else False
        """
        return_val = True
        try:
            list_of_udf_profile_metadata = self.model.query.filter(
                self.model._udf_id == udf_id
            )
            for udf_profile_metadata in list_of_udf_profile_metadata:
                udf_profile_metadata.delete()
        except Exception:
            logger.exception("Delete udf Profile failed for id={}".format(udf_id))
            return_val = False

        self.print_all_profile("Post Dropping {}".format(udf_id))
        return return_val

    def udf_within_deadline(self, list_of_udf_ids, time, cardinality):
        udfs = []
        for (udf_id, udf_name, impl_file_path, accuracy) in list_of_udf_ids:
            func = path_to_class(impl_file_path, udf_name)()
            selected_profiles = self.model.query.filter(self.model._udf_id == udf_id).filter(
                self.model._time_taken * cardinality / self.model._batch_size <= time).all()
            for profile in selected_profiles:
                udfs.append((udf_name, profile._time_taken * cardinality / profile._batch_size, accuracy, func))
        return udfs

    def print_all_profile(self, when=""):
        print(when)
        list_of_udf_profile_metadata = self.get_all_profiles()
        for udf_profile_metadata in list_of_udf_profile_metadata:
            print("  {} {} {} {}".format(
                udf_profile_metadata._udf_id,
                udf_profile_metadata._id,
                udf_profile_metadata._batch_size,
                udf_profile_metadata._time_taken,
            ))

    def get_all_profiles(self):
        try:
            return self.model.query.all()
        except NoResultFound:
            return []