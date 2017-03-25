""" Please call the file in root fold
    python utils/draw_test.py
"""

import sys
sys.path.append('.')

import utils.draw as draw
import utils.tfevents as tfevents


# get specify data
tags, iter_tag = tfevents.get_tags('regression', 'train')

# parse data
values = tfevents.get_image_summary(
    '_output/cifar_train_201703251903', tags, iter_tag)

# transfer data to draw
info = tfevents.get_info_data(values, tags, iter_tag)

# draw data
draw.draw(info, tags, iter_tag)
