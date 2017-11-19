# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A variety of RNN implement"""

import tensorflow as tf

from gate.rnn.brnn import brnn
from gate.rnn.basic_rnn import basic_rnn

networks_map = {
    'brnn': brnn,
    'basic_rnn': basic_rnn
}


def get_network(X, rnn_cfg, audio_cfg, name_scope='', reuse=None):
    """ A factory to group the rnn network
    """
    if rnn_cfg.net_name not in networks_map:
        raise ValueError('Unknown network name %s' % rnn_cfg.net_name)
    net = networks_map[rnn_cfg.net_name](rnn_cfg)
    with tf.variable_scope(name_scope + rnn_cfg.net_name) as scope:
        if reuse:
            scope.reuse_variables()
        return net.model(X, audio_cfg)
