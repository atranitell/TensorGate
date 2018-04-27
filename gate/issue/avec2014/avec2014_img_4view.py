# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/2/25

--------------------------------------------------------

FOR AVEC2014

"""

import tensorflow as tf
from gate import context
from gate.net.factory import get_net
from gate.data.factory import get_data
from gate.solver import updater
from gate.layer import l2
from gate.util import variable
from gate.util import filesystem
from gate.util import string
from gate.util.logger import logger
from gate.issue.avec2014.utils import get_accurate_from_file


class AVEC2014_IMG_4VIEW(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)