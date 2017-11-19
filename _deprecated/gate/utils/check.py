# -*- coding: utf-8 -*-
""" updated: 2017/3/28

    For check input content
"""
from gate.utils.logger import logger


def raise_none_param(*config):
    """ Check input if none
    """
    for arg in config:
        if arg is None:
            logger.error(config)
            raise ValueError('Input is None type, Please check again.')
