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
    Configbase.__init__(self, args)
    self.name = 'avec2014'
    self.target = 'avec2014.img.cnn'
    self.output_dir = None
    self.task = 'train'

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


class AVEC2014_DLDL(Configbase):

  def __init__(self, args):
    """AVEC2014 dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014.dldl'
    self.target = 'avec2014.img.cnn.dldl'
    self.output_dir = None
    self.task = 'train'

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
        num_classes=64,
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
    data.set_label(num_classes=64, span=64, one_hot=False, scale=False)


class AVEC2014_EX1(Configbase):

  def __init__(self, args):
    """AVEC2014 dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014.ex1'
    self.target = 'avec2014.img.cnn.ex1'
    self.output_dir = None
    self.task = 'train'

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
        num_classes=64,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """learning rate"""
    self.lr = [params.LR(), params.LR()]
    self.lr[0].set_fixed(learning_rate=0.001)
    self.lr[1].set_fixed(learning_rate=0.0001)

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_img_with_c.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=32)
    self.set_train_data_attr(self.train.data)

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

  def set_train_data_attr(self, data):
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
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=64, span=64, one_hot=False, scale=False)

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
    data.set_label(num_classes=64, span=64, one_hot=False, scale=False)


class AVEC2014_68(Configbase):

  def __init__(self, args):
    """AVEC2014 dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014.68'
    self.target = 'avec2014.68.img.cnn'
    self.output_dir = None
    self.task = 'train'

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
    self.net[0].resnet_v2_critical(
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
        entry_path='../_datasets/AVEC2014/pp_trn_0_img_aligned_int.txt',
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
        entry_path='../_datasets/AVEC2014/pp_tst_img_aligned_int.txt',
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
        output_height=256,
        output_width=256,
        preprocessing_method='avec2014.68',
        gray=False)
    data.add(image)
    data.set_entry_attr(
        tuple([str, ]+[float, float, float]*68+[float, ]),
        tuple([True, ]+[False]*205))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_FLOW(Configbase):

  def __init__(self, args):
    """AVEC2014_FLOW dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014.flow'
    self.target = 'avec2014.img.cnn'
    self.output_dir = None
    self.task = 'train'

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
    Configbase.__init__(self, args)
    self.name = 'avec2014'
    self.target = 'avec2014.img.bicnn.shared'
    self.output_dir = None
    self.task = 'train'

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


class AVEC2014_AUDIO_CNN(Configbase):

  def __init__(self, args):
    """AVEC2014_AUDIO dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    r = self._read_config_file

    self.name = r('avec2014', 'name')
    self.target = r('avec2014.audio.cnn', 'target')
    self.output_dir = r(None, 'output_dir')
    self.ckpt_file = r(None, 'ckpt_file')
    self.task = r('train', 'task')

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=150000)

    """network model"""
    self.net = [params.NET()]
    self.net[0].sensnet(
        num_classes=1,
        weight_decay=0.0001,
        unit_type=r('multi_addition', 'net.unit_type'),
        unit_num=r([1, 1, 1, 1], 'net.unit_num'),
        batch_norm_decay=0.999,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=r(False, 'net.use_batch_norm'),
        use_pre_batch_norm=r(False, 'net.use_pre_batch_norm'),
        dropout_keep=0.5,
        activation_fn=r('leaky_relu', 'net.activation_fn'),
        version=r('sensnet_v2_fpn', 'net.version'))

    """learning rate"""
    self.lr = [params.LR()]
    self.lr[0].set_fixed(learning_rate=r(0.00003, 'train.lr'))

    """optimizer"""
    self.opt = [params.OPT()]
    self.opt[0].set_adam()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=r(32, 'train.batchsize'),
        entry_path=r('../_datasets/AVEC2014_Audio/pp_trn_raw.txt',
                     'train.entry_path'),
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """val"""
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=r(50, 'val.batchsize'),
        entry_path=r('../_datasets/AVEC2014_Audio/pp_val_16.txt',
                     'val.entry_path'),
        shuffle=False)
    self.val.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.val.data)

    """test"""
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=r(50, 'test.batchsize'),
        entry_path=r('../_datasets/AVEC2014_Audio/pp_tst_16.txt',
                     'test.entry_path'),
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

    """extract_feature"""
    self.extract_feature = params.Phase('extract_feature')
    self.extract_feature.data = params.DATA(
        batchsize=r(50, 'extract_feature.batchsize'),
        entry_path=r('../_datasets/AVEC2014_Audio/pp_tst_16.txt',
                     'extract_feature.entry_path'),
        shuffle=False)
    self.extract_feature.data.set_queue_loader(
        loader='load_audio',
        reader_thread=1,  # keep order
        min_queue_num=128)
    self.set_default_data_attr(self.extract_feature.data)

  def set_default_data_attr(self, data):
    r = self._read_config_file
    audio = params.Audio()
    audio.set_fixed_length_audio(
        frame_num=r(16, 'audio.frame_num'),
        frame_length=r(200, 'audio.frame_length'),
        frame_invl=r(200, 'audio.frame_invl'))
    data.add(audio)
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_AUDIO_FCN(Configbase):

  def __init__(self, args):
    """AVEC2014_AUDIO dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014'
    self.target = 'avec2014.audio.fcn'
    self.output_dir = None
    self.task = 'train'

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

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
        entry_path='../_datasets/AVEC2014_Audio/pp_val_16.txt',
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
        entry_path='../_datasets/AVEC2014_Audio/pp_tst_16.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_audio',
        reader_thread=8,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    audio = params.Audio()
    audio.set_fixed_length_audio(
        frame_num=16, frame_length=200, frame_invl=200)
    data.add(audio)
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1, span=63, one_hot=False, scale=True)


class AVEC2014_AUDIO_VAE(Configbase):

  def __init__(self, args):
    """AVEC2014_AUDIO dataset for Depressive Detection"""
    Configbase.__init__(self, args)
    self.name = 'avec2014'
    self.target = 'avec2014.audio.vae'
    self.output_dir = None
    self.task = 'train'

    """iteration controller"""
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=5000,
        val_invl=5000,
        max_iter=1500000)

    """learning rate"""
    self.lr = [params.LR(), params.LR()]
    self.lr[0].set_fixed(learning_rate=0.00001)
    self.lr[1].set_fixed(learning_rate=0.00003)

    """optimizer"""
    self.opt = [params.OPT(), params.OPT()]
    self.opt[0].set_rmsprop()
    self.opt[1].set_rmsprop()

    """train"""
    self.train = params.Phase('train')
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014_Audio/pp_trn_raw.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_pair_audio',
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
