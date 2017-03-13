
from abc import ABCMeta, abstractmethod

class Net(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def arg_scope(self):
        pass