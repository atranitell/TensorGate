
from gate.utils import string
from gate.utils.logger import logger


class Database():
    """ a factory combined all components.
    """

    def __init__(self):
        pass

    def loads(self):
        pass

    def _print(self):
        if self.data_type == 'train':
            logger.info(string.class_members(self.log))

            logger.info('Total num: %d, batch size: %d, height: %d, width: %d, channels: %d'
                        % (self.total_num, self.batch_size, self.output_height, self.output_width, self.channels))

            if self.preprocessing_method is not None:
                logger.info('preprocessing method: %s' %
                            self.preprocessing_method)

            if self.opt is not None:
                logger.info(string.class_members(self.opt))

            if self.lr is not None:
                logger.info(string.class_members(self.lr))

            if self.hps is not None:
                logger.info(string.class_members(self.hps))
