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
"""Snapshot"""
import os
import tensorflow as tf
from gate.utils.logger import logger


class Snapshot():

  def __init__(self, config):
    self.config = config
    self.chkp_hook = None
    self.hook = None
    self.name = config.name

  def init(self):
    """Should be init after updater"""
    self.hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=self.config.output_dir,
        save_steps=self.config.log.save_model_invl,
        saver=tf.train.Saver(
            var_list=tf.global_variables(), max_to_keep=10000),
        checkpoint_basename=self.name + '.ckpt')
    return self.hook

  def restore(self, sess, saver):
    """If the ckpt_file exists, the program will prefer to load it.
      else load the output_dir
    """
    if self.config.ckpt_file is not None:
      ckpt = True
      ckpt_path = os.path.join(self.config.output_dir, self.config.ckpt_file)
    else:
      ckpt = tf.train.get_checkpoint_state(self.config.output_dir)
      ckpt_path = ckpt.model_checkpoint_path
    if ckpt is not None and ckpt_path:
      saver.restore(sess, ckpt_path)
      global_step = ckpt_path.split('/')[-1].split('-')[-1]
      try:
        int(global_step)
      except:
        global_step = 0
      logger.sys('Load checkpoint from: %s' % ckpt_path)
      return global_step
    else:
      logger.sys('Start a new model to train.')
      return
