# -*- coding: utf-8 -*-
""" updated: 2017/04/05
    In order to control output information,
        All print will be wrapped here.
    We could specify some kind of output.
"""
from datetime import datetime

# add date
_DATE = True

_SYS = True
_TRAIN = True
_TEST = True
_WARN = True
_INFO = True
_NET = True


def _print(show_type, content):
    """ format print string
    """
    if _DATE:
        str_date = '[' + datetime.strftime(datetime.now(), '%y.%m.%d %H:%M:%S') + '] '
        print(str_date + show_type + ' ' + content)
    else:
        print(show_type + ' ' + content)


def SYS(content):
    """ Print information related to build system.
    """
    if _SYS:
        _print('[SYS]', content)


def NET(content):
    """ build net graph related infomation.
    """
    if _NET:
        _print('[NET]', content)


def TRAIN(content):
    """ relate to the training processing.
    """
    if _TRAIN:
        _print('[TRN]', content)


def TEST(content):
    """ relate to the test processing.
    """
    if _TEST:
        _print('[TST]', content)


def WARN(content):
    """ some suggest means warning.
    """
    if _WARN:
        _print('[WAN]', content)


def INFO(content):
    """ just print it for check information
    """
    if _INFO:
        _print('[INF]', content)


def type_list_to_str(dtype_list):
    """ (type1, type2, type3)
    """
    return '(' + ', '.join([item.__name__ for item in dtype_list]) + ')'


def class_members(obj):
    INFO(', '.join(['%s: %s' % item for item in sorted(obj.__dict__.items())]))
