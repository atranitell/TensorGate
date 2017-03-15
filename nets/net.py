
from abc import ABCMeta, abstractmethod

class Net(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def model(self, images, num_classes, is_training):
        pass

    @abstractmethod
    def arg_scope(self):
        pass