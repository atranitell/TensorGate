# -*- coding: utf-8 -*-
import os
import py_compile

DIR_BIN = '_bin'


def removeDir(dirPath):
    if not os.path.isdir(dirPath):
        return
    files = os.listdir(dirPath)
    for file in files:
        filePath = os.path.join(dirPath, file)
        if os.path.isfile(filePath):
            os.remove(filePath)
        elif os.path.isdir(filePath):
            removeDir(filePath)
    os.rmdir(dirPath)


def process(root, fname):
    src = os.path.join(root, fname)
    dst = src.replace('./', DIR_BIN + '/') + 'c'
    py_compile.compile(src, cfile=dst, optimize=2)


def compile():
    # remove old
    if os.path.exists(DIR_BIN):
        removeDir(DIR_BIN)
    if not os.path.exists(DIR_BIN):
        os.mkdir(DIR_BIN)

    for _i in os.walk('./'):
        fold = _i[0]
        for fname in _i[2]:
            ext = fname.split('.')
            if len(ext) > 1 and ext[1] == 'py':
                process(fold, fname)


if __name__ == '__main__':
    compile()
