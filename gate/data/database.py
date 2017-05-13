
from gate.utils import show


class Database():
    """ a factory combined all components.
    """

    def __init__(self):
        pass

    def loads(self):
        pass

    def _print(self):
        if self.data_type == 'train':
            show.class_members(self.log)
            show.INFO('Total num: %d, batch size: %d, height: %d, width: %d, channels: %d'
                      % (self.total_num, self.batch_size,
                         self.output_height, self.output_width, self.channels))
            if self.preprocessing_method is not None:
                show.INFO('preprocessing method: %s' %
                          self.preprocessing_method)
            if self.opt is not None:
                show.class_members(self.opt)
            if self.lr is not None:
                show.class_members(self.lr)
