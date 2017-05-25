
import logging
from datetime import datetime


class Logger():

    def __init__(self):
        self.logger = logging.getLogger('TensorGate')
        self.logger.setLevel(logging.DEBUG)
        self._DATE = True
        self._SYS = True
        self._TRAIN = True
        self._TEST = True
        self._VAL = True
        self._NET = True
        self._WARN = True
        self._INFO = True
        self._ERR = True

    def set_filestream(self, filepath, level=logging.DEBUG):
        """ setting content output to file
        """
        fh = logging.FileHandler(filepath)
        fh.setLevel(level)
        self.logger.addHandler(fh)

    def set_screenstream(self, level=logging.DEBUG):
        """ setting content output to screen
        """
        ch = logging.StreamHandler()
        ch.setLevel(level)
        self.logger.addHandler(ch)

    def _print(self, show_type, content):
        """ format print string
        """
        if self._DATE:
            str_date = '[' + \
                datetime.strftime(datetime.now(), '%y.%m.%d %H:%M:%S') + '] '
            self.logger.info(str_date + show_type + ' ' + content)
        else:
            self.logger.info(show_type + ' ' + content)

    def sys(self, content):
        """ Print information related to build system.
        """
        if self._SYS:
            self._print('[SYS]', content)

    def net(self, content):
        """ build net graph related infomation.
        """
        if self._NET:
            self._print('[NET]', content)

    def train(self, content):
        """ relate to the training processing.
        """
        if self._TRAIN:
            self._print('[TRN]', content)

    def val(self, content):
        """ relate to the validation processing.
        """
        if self._TRAIN:
            self._print('[VAL]', content)

    def test(self, content):
        """ relate to the test processing.
        """
        if self._TEST:
            self._print('[TST]', content)

    def warn(self, content):
        """ some suggest means warning.
        """
        if self._WARN:
            self._print('[WAN]', content)

    def info(self, content):
        """ just print it for check information
        """
        if self._INFO:
            self._print('[INF]', content)

    def error(self, content):
        """ For error info
        """
        if self._ERR:
            self._print('[ERR]', content)


logger = Logger()
