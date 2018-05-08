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
"""TRAFFICFLOW DATASET"""

import gate.config.config_params as params
from gate.config.config_base import Configbase


class TRAFFICFLOW(Configbase):

  def __init__(self, args):
    """TRAFFICFLOW dataset"""
    self.name = 'trafficflow'
    self.target = 'trafficflow.vanilla'
    self.output_dir = None
    self.task = 'train'
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=100000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].resnet_v2(
        depth='50',
        num_classes=1,
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True)

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(0.0001)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/trafficflow/train.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_npy',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=100,
        entry_path='../_datasets/trafficflow/test.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_npy',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    data.add(params.Numpy([112, 112, 10]))
    data.set_entry_attr((str, float), (True, False))
    data.set_label(num_classes=1)
