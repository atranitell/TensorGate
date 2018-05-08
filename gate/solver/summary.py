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
"""Summary"""

import tensorflow as tf
from gate.utils.logger import logger


class Summary():

  def __init__(self, config):
    """ summary helper
    """
    # setting path
    self.config = config
    self.hook = None
    self.summary = tf.summary.FileWriter(self.config.output_dir)

  def init(self):
    self.hook = tf.train.SummarySaverHook(
        save_steps=self.config.log.save_summary_invl,
        output_dir=self.config.output_dir,
        summary_op=tf.summary.merge_all())
    return self.hook

  def add(self, tag, value):
    """ add a simple value to summary
    """
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    self.summary.add_summary(summary)

  def adds(self, global_step, tags, values):
    """ add a series of values
    """
    summary = tf.Summary()
    for i, v in enumerate(tags):
      summary.value.add(tag=tags[i], simple_value=values[i])
    self.summary.add_summary(summary, global_step=global_step)
