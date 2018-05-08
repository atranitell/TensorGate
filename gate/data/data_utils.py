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
"""Data UTILS"""

import tensorflow as tf


def convert_to_tensor(res_list, type_list):
  """Convert python in-built type to tensor type."""
  tensors = []
  for i, t in enumerate(type_list):
    if t == str:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.string)
    elif t == int:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.int32)
    elif t == float:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.float32)
    else:
      raise ValueError('Unknown Input Type [%s]' % t)
    tensors.append(r)
  return tensors
