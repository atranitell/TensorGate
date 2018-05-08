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
"""AVEC2014 DATASET
- avec2014: normal single image
- avec2014.flow: normal single optical flow image
- avec2014.bicnn: .normal/.rgb/.opt/.shared
"""

import gate.config.config_params as params
from gate.config.config_base import Configbase


class AVEC2014(Configbase):

  def __init__(self, args):
    """AVEC2014 dataset for Depressive Detection"""
    self.name = 'avec2014'
    self.target = 'avec2014.img.cnn'
    self.output_dir = None
    self.task = 'train'
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].resnet_v2(
        depth='50',
        num_classes=1,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(learning_rate=0.001)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_img.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_img.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

    """ data.heatmap """
    self.heatmap = params.Phase('heatmap')
    self.heatmap.data = params.DATA(
        batchsize=1,
        entry_path='../_datasets/AVEC2014/pp_tst_img_paper.txt',
        shuffle=False)
    self.heatmap.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=1)
    self.set_default_data_attr(self.heatmap.data)

  def set_default_data_attr(self, data):
    image = params.Image()
    image.set_fixed_length_image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.add(image)
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_FLOW(Configbase):

  def __init__(self, args):
    """AVEC2014_FLOW dataset for Depressive Detection"""
    self.name = 'avec2014.flow'
    self.target = 'avec2014.img.cnn'
    self.output_dir = None
    self.task = 'train'
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].resnet_v2(
        depth='50',
        num_classes=1,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(learning_rate=0.001)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_flow.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_flow.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    image = params.Image()
    image.set_fixed_length_image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.add(image)
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_BICNN(Configbase):

  def __init__(self, args):
    """AVEC2014_BICNN dataset for Depressive Detection"""
    self.name = 'avec2014'
    self.target = 'avec2014.img.bicnn.shared'
    self.output_dir = None
    self.task = 'train'
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].resnet_v2_bishared(
        depth='50',
        num_classes=1,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(learning_rate=0.001)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_img_flow.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_img_flow.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    image = params.Image()
    image.set_fixed_length_image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.add([image, image])
    data.set_entry_attr((str, str, int), (True, True, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_AUDIO(Configbase):

  def __init__(self, args):
    """AVEC2014_AUDIO dataset for Depressive Detection"""
    self.name = 'avec2014'
    self.target = 'avec2014.audio.cnn'
    self.output_dir = None
    self.task = 'train'
    # rewrite default setting
    super().__init__(args)

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].audionet(
        num_classes=1,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(learning_rate=0.00003)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014_Audio/pp_trn_raw.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """val"""
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014_Audio/pp_val_32.txt',
        shuffle=False)
    self.val.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.val.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014_Audio/pp_tst_32.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    audio = params.Audio()
    audio.set_fixed_length_audio(
        frame_num=32, frame_length=200, frame_invl=200)
    data.add(audio)
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)