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
"""Data Processing"""

import tensorflow as tf

from gate.processing.slim import cifarnet_preprocessing
from gate.processing.slim import avec2014_preprocessing
from gate.processing.slim import inception_preprocessing
from gate.processing.slim import lenet_preprocessing
from gate.processing.slim import vgg_preprocessing
from gate.processing.slim import kinship_vae_preprocessing


preprocessing_map = {
    'cifarnet': cifarnet_preprocessing,
    'inception': inception_preprocessing,
    'lenet': lenet_preprocessing,
    'vgg': vgg_preprocessing,
    'vae.kinship': kinship_vae_preprocessing,
    'avec2014': avec2014_preprocessing
}


def get_preprocessing(X, name, cfg, phase):
  is_training = True if phase == 'train' else False
  with tf.name_scope('preprocessing/' + name):
    return preprocessing_map[name].preprocess_image(
        X, cfg.output_height, cfg.output_width, is_training)
