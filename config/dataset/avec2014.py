# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2018-02-05
"""
from core.data import database
from core.data import data_params as params


class AVEC2014(database.DatasetBase):
  """ Based on images extracted from the video frames.
  """

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'avec2014'
    self.target = 'avec.image.cnn'
    self.data_dir = '../_datasets/AVEC2014'
    self.task = 'heatmap'
    self.output_dir = '../_model/avec2014_resnet_50'
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    """ net """
    self.net = params.Net('resnet_v2_50')
    self.net.set_weight_decay(0.0005)
    self.net.set_batch_norm(
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam(beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-8)
    self.train.data = params.Data(
        batchsize=32,
        entry_path="pp_trn_0_img.txt",
        shuffle=True,
        total_num=23564,
        loader='load_image',
        reader_thread=16)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=1,
        entry_path="pp_tst_img_paper.txt",
        shuffle=False,
        total_num=10,  # 17727,
        loader='load_image',
        reader_thread=16)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img])
    data.set_entry_attr(
        entry_dtype=(str, int),
        entry_check=(True, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
    return data


class AVEC2014_AUDIO(database.DatasetBase):

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'avec2014.audio'
    self.target = 'avec.audio.cnn'
    self.data_dir = '../_datasets/AVEC2014_Audio'
    self.task = 'train'
    self.output_dir = None  # 'C:/Users/KaiJIN/Desktop/Gate/SensNet-v2/visualization'
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=50,
        save_summaries_invl=50,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    """ net """
    self.net = params.Net('audionet')
    self.net.set_weight_decay(0.0001)
    self.net.set_dropout_keep(0.5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.00003)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam()

    self.train.data = params.Data(
        batchsize=32,
        entry_path="pp_trn_raw.txt",
        shuffle=True,
        total_num=100,
        loader='load_audio',
        reader_thread=32)
    self.train.data = self.set_data_attr(self.train.data)

    """ val """
    self.val = params.Phase('val')
    self.val.data = params.Data(
        batchsize=50,
        entry_path="pp_val_32.txt",  # 32
        shuffle=False,
        total_num=14199,  # 14199-32 28542-16 13999-64
        loader='load_audio',
        reader_thread=1)
    self.val.data = self.set_data_attr(self.val.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="pp_tst_32.txt",  # 32
        shuffle=False,
        total_num=25465,  # 25465-32 51074-16 25265-64
        loader='load_audio',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_audio = params.Audio()
    default_audio.frame_num = 32  # 32
    default_audio.frame_length = 200
    default_audio.frame_invl = 200
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_audio([default_audio])
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
    return data


class AVEC2014_BI(database.DatasetBase):
  """ RGB + Optical Flow
  """

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'avec2014'
    self.target = 'avec.image.bicnn'
    self.data_dir = '../_datasets/AVEC2014'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    """ net """
    self.net = params.Net('resnet_v2_50')
    self.net.set_weight_decay(0.0005)
    self.net.set_batch_norm(
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam(beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-8)
    self.train.data = params.Data(
        batchsize=32,
        entry_path="pp_trn_0_img_flow.txt",
        shuffle=True,
        total_num=23564,
        loader='load_image',
        reader_thread=16)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="pp_tst_img_flow.txt",
        shuffle=False,
        total_num=17727,
        loader='load_image',
        reader_thread=16)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img, default_img])
    data.set_entry_attr(
        entry_dtype=(str, str, int),
        entry_check=(True, True, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
    return data


class AVEC2014_BISHARED(database.DatasetBase):
  """ RGB + Optical Flow Shared base layer
  """

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'avec2014'
    self.target = 'avec.image.bishared'
    self.data_dir = '../_datasets/AVEC2014'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    """ net """
    self.net = params.Net('resnet_v2_50_bishared')
    self.net.set_weight_decay(0.0005)
    self.net.set_batch_norm(
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam(beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-8)
    self.train.data = params.Data(
        batchsize=32,
        entry_path="pp_trn_0_img_flow.txt",
        shuffle=True,
        total_num=23564,
        loader='load_image',
        reader_thread=16)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="pp_tst_img_flow.txt",
        shuffle=False,
        total_num=17727,
        loader='load_image',
        reader_thread=16)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=256,
        raw_width=256,
        output_height=224,
        output_width=224,
        preprocessing_method='avec2014',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img, default_img])
    data.set_entry_attr(
        entry_dtype=(str, str, int),
        entry_check=(True, True, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
    return data
