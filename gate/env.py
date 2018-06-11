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
"""ENVIRONMENT FOR COMMON CALLING"""

from gate.utils import filesystem


class Env():
  """Environment Variables."""

  _OUTPUT = filesystem.mkdir('../_outputs/')
  _DATASET = '../_datasets'

  # logger config
  _LOG_DATE = True
  _LOG_SYS = False
  _LOG_TRAIN = True
  _LOG_TEST = True
  _LOG_VAL = True
  _LOG_NET = True
  _LOG_WARN = True
  _LOG_INFO = True
  _LOG_ERR = True
  _LOG_CFG = True
  _LOG_TIMER = True

  # SUMMARY SCALAR
  _SUMMARY_SCALAR = True
  # SUMMARY SETTING
  _SUMMARY_GRAD_STAT = True
  _SUMMARY_GRAD_HIST = True
  _SUMMARY_WEIGHT_STAT = True
  _SUMMARY_WEIGHT_HIST = True


env = Env()
