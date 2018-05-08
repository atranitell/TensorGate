# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
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
# ==============================================================================
"""Data Loader"""

from gate.data.tfqueue import data_loader
# from gate.data.custom import db_coco

loader_maps = {
    'load_image': data_loader.load_image,
    'load_npy': data_loader.load_npy,
    'load_audio': data_loader.load_audio,
    # 'coco': db_coco.DB_COCO
}


def load_data(config):
  return loader_maps[config.data.loader](config)
