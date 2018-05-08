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
"""KINFACE DATASET"""

import gate.config.config_params as params
from gate.config.config_base import Configbase


class KinfaceVAE(Configbase):

  def __init__(self, args):
    """Kinface dataset for classification"""
    r = self._read_config_file
    self.name = r('kinface2.vae', 'name')
    self.target = r('kinface.1E1G1D', 'target')
    self.output_dir = r(None, 'output_dir')
    self.task = r('train', 'task')
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=15000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].kinvae(z_dim=100)

    """train.lr"""
    self.lr = [params.LR(), params.LR()]
    self.lr[0].set_fixed(learning_rate=r(0.000005, 'train.lr0'))
    self.lr[1].set_fixed(learning_rate=r(0.00005, 'train.lr1'))
    """train.opt"""
    self.opt = [params.OPT(), params.OPT()]
    self.opt[0].set_rmsprop()
    self.opt[1].set_rmsprop()

    """ train """
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=r(32, 'train.batchsize'),
        entry_path=r('../_datasets/kinface2/train_kinvae_1.txt', 'train.entry_path'),
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ val """
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=r(100, 'test.batchsize'),
        entry_path=r('../_datasets/kinface2/train_kinvae_1.txt', 'val.entry_path'),
        shuffle=False)
    self.val.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.val.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=r(100, 'test.batchsize'),
        entry_path=r('../_datasets/kinface2/test_kinvae_1.txt', 'test.entry_path'),
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    image = params.Image()
    image.set_fixed_length_image(
        channels=3,
        frames=1,
        raw_height=64,
        raw_width=64,
        output_height=64,
        output_width=64,
        preprocessing_method='vae.kinship',
        gray=False)
    data.add([image, image, image, image])
    data.set_entry_attr(
      (str, str, str, str, int, int),
      (True, True, True, True, False, False))
    data.set_label(num_classes=4)
