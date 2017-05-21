# -*- coding: utf-8 -*-
""" string operation
"""


def clip_last_sub_string(string, separator='/', keep_sep=False):
    """ raw: a/b/c/d/e
        return: a/b/c/d/
    """
    st = str(string).split(separator)
    nw = ''
    for i in range(len(st) - 1):
        if keep_sep is True and i == len(st) - 2:
            nw += st[i]
        else:
            nw += st[i] + separator
    return nw


def type_list_to_str(dtype_list):
    """ (type1, type2, type3)
    """
    return '(' + ', '.join([item.__name__ for item in dtype_list]) + ')'


def class_members(obj):
    return ', '.join(['%s: %s' % item for item in sorted(obj.__dict__.items())])
