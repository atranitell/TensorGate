# -*- coding: utf-8 -*-
# ./data/datasets_data_provider.py
#
#    Tensorflow Version: r1.0
#    Python Version: 3.5
#    Update Date: 2017/03/13
#    Author: Kai JIN
# ==============================================================================

""" Contains the definition of a Dataset.

Functions:
    - images and labels preprocessing
    - from file/folder to load images and labels
    - generate parallel queue
    - batch for training or testing

Constant:
    - the total number of images
    - the input type 'train'/'test' in according to the different preprocessing method
    - datasets name
    - batch size

Call Method:
    main->datasets_factory->(cifar10/..)->dataset
"""

from abc import ABCMeta, abstractmethod

class Dataset(metaclass=ABCMeta):
    """ Represents a Dataset specification.

    Note:
        any datasets('cifar','imagenet', and etc.) should inherit this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def loads(self):
        pass
    
    @abstractmethod
    def _generate_image_label_batch(self):
        pass
    
    @abstractmethod
    def _preprocessing_image(self):
        pass
    
    @abstractmethod
    def _preprocessing_label(self):
        pass