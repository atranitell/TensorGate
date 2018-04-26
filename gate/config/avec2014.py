# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/26

--------------------------------------------------------

AVEC2014 Dataset

- avec2014: normal single image
- avec2014.flow: normal single optical flow image
- avec2014.bi: image + optical flow
- avec2014.bishared: image + optical flow (deep fuse)

"""

from gate.config import base
from gate.config import params


class AVEC2014(base.ConfigBase):

  def __init__(self, config):

    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'avec2014'
    self.target = 'avec2014.img.cnn'
    self.data_dir = '../_datasets/AVEC2014'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """ net """
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

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_adam(beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-8)

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_img.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_img.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
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
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)


class AVEC2014_FLOW(base.ConfigBase):

  def __init__(self, config):

    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'avec2014.flow'
    self.target = 'avec2014.img.cnn'
    self.data_dir = '../_datasets/AVEC2014'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """ net """
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

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_adam(beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-8)

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_flow.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_flow.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
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
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)


class AVEC2014_BICNN(base.ConfigBase):

  def __init__(self, config):

    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'avec2014.bicnn'
    # .normal/.rgb/.opt/.shared
    self.target = 'avec2014.img.bicnn.shared'
    self.data_dir = '../_datasets/AVEC2014'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """ net """
    self.net = [params.NET()]
    # self.net[0].resnet_v2_bishared(
    #     depth='50',
    #     num_classes=1,
    #     weight_decay=0.0005)

    self.net[0].vgg_bishared(
        depth='11',
        num_classes=1,
        weight_decay=0.0005,
        global_pool=True)

    # self.net[0].alexnet_bishared(
    #     num_classes=1,
    #     weight_decay=0.0005,
    #     global_pool=True)

    # self.net = [params.NET(), params.NET()]
    # self.net[0].resnet_v2(
    #     depth='50'
    #     num_classes=1,
    #     weight_decay=0.0005)
    # self.net[1].resnet_v2(
    #     depth='50'
    #     num_classes=1,
    #     weight_decay=0.0005)

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_adam(beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-8)

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014/pp_trn_0_img_flow.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014/pp_tst_img_flow.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=16,
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
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)


class AVEC2014_AUDIO(base.ConfigBase):

  def __init__(self, config):

    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'avec2014'
    self.target = 'avec2014.audio.cnn'
    self.data_dir = '../_datasets/AVEC2014_Audio'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """ net """
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

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(learning_rate=0.00003)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_adam()

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/AVEC2014_Audio/pp_trn_raw.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_audio',
        reader_thread=16,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.val """
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014_Audio/pp_val_32.txt',
        shuffle=False)
    self.val.data.set_queue_loader(
        loader='load_audio',
        reader_thread=16,
        min_queue_num=50)
    self.set_default_data_attr(self.val.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=50,
        entry_path='../_datasets/AVEC2014_Audio/pp_tst_32.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_audio',
        reader_thread=16,
        min_queue_num=50)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    audio = params.Audio()
    audio.set_fixed_length_audio(
        frame_num=32,
        frame_length=200,
        frame_invl=200)
    data.add(audio)
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
