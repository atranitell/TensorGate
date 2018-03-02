# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2018-02-05
"""
from core.data import database
from core.data import data_params as params


class AVEC2014_AUDIO(database.DatasetBase):

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'avec2014.audio'
    self.target = 'avec.audio.cnn'
    self.data_dir = '../_datasets/AVEC2014_Audio'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=1,
        save_summaries_invl=10,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    """ net """
    self.net = params.Net('audionet')
    self.net.set_weight_decay(0.0001)
    # self.net.set_dropout_keep(0.5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam()
    self.train.data = params.Data(
        batchsize=32,
        entry_path="pp_trn_succ64.txt",
        shuffle=True,
        total_num=15292,
        loader='load_audio',
        reader_thread=32)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="pp_tst_succ64.txt",
        shuffle=False,
        total_num=25465,
        loader='load_audio',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_audio = params.Audio()
    default_audio.length = 6400
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_audio([default_audio])
    data.set_entry_attr((str, int, int), (True, False, False))
    data.set_label(num_classes=1,
                   span=63,
                   one_hot=False,
                   scale=True)
    return data
