# -*- coding: utf-8 -*-
""" showing the progressive bar
"""

import sys


class Progressive():

    def __init__(self, min_scale=1.0):
        sys.stdout.write('[TEST] RUN [')
        tab = ''
        self.min_scale = min_scale
        self.num_partion = int(100 / self.min_scale)
        for _ in range(self.num_partion + 4):
            tab += ' '
        sys.stdout.write(tab + ']')
        self.cur = 0

    def add(self):
        self.cur += 1
        if self.cur > self.num_partion:
            return

        for _ in range(self.num_partion + 6 - self.cur):
            sys.stdout.write('\b')

        sys.stdout.write('#')

        for _ in range(self.num_partion - self.cur):
            sys.stdout.write(' ')

        c = self.cur * self.min_scale
        sys.stdout.write('] %2d%%' % c)

        sys.stdout.flush()
        # time.sleep(0.01)

    def add_float(self, cur, max_cur):
        # print(self.min_scale*self.cur)
        if int(cur * 100 / max_cur) > self.min_scale * self.cur:
            self.add()
        else:
            return
