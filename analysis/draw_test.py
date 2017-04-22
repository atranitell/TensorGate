""" Please call the file in root fold
    python utils/draw_test.py
"""

import sys
sys.path.append('.')

import analysis.draw as draw
import analysis.data_tfevents as tfevents


# get specify data
tags, iter_tag = tfevents.get_tags('regression', 'train')

# parse data
values = tfevents.get_image_summary(
    '../_output/MODEL/avec2014_train_inception_resnet_v2_fold0', tags, iter_tag)

# transfer data to draw
info = tfevents.get_info_data(values, tags, iter_tag)

# draw data
draw.draw(info, tags, iter_tag)
