
from gate.utils import string
from gate.utils.logger import logger


class Database():
    """ a factory combined all components.
    """

    def __init__(self):
        """ basic info"""
        self.image = None
        self.audio = None
        self.rnn = None
        self.lr = None
        self.opt = None
        self.hps = None
        self.data_type = None
        self.log = None
        self.total_num = None
        self.batch_size =None

    def loads(self):
        """ data loader
        """
        pass

    def _set_phase(self, data_type):
        """ for different phase use differnt data_path and shuffle
            train for config lr and optimizer
        """
        if data_type == 'train':
            self._train()
        elif data_type == 'val':
            self._val()
        elif data_type == 'test':
            self._test()
        elif data_type == 'val_train':
            self._val_train()
        else:
            raise ValueError('Unknown command %s' % data_type)

    def _train(self):
        raise ValueError('Function Should defined in inherited class.')

    def _val_train(self):
        raise ValueError('Function Should defined in inherited class.')

    def _val(self):
        raise ValueError('Function Should defined in inherited class.')

    def _test(self):
        raise ValueError('Function Should defined in inherited class.')

    def _print(self):
        if self.data_type == 'train':
            logger.info(string.class_members(self.log))
            logger.info('Total num: %d, batch size: %d.' %
                        (self.total_num, self.batch_size))
            if self.image is not None:
                logger.info(string.class_members(self.image))
            if self.audio is not None:
                logger.info(string.class_members(self.audio))
            if self.rnn is not None:
                logger.info(string.class_members(self.rnn))
            if self.opt is not None:
                logger.info(string.class_members(self.opt))
            if self.lr is not None:
                logger.info(string.class_members(self.lr))
            if self.hps is not None:
                logger.info(string.class_members(self.hps))
