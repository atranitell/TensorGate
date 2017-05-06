""" Please call the file in root fold
    python utils/draw_test.py
"""

import tfevents_draw
import tfevents


# get specify data
tags, iter_tag = tfevents.get_tags('regression', 'train')

# parse data
values = tfevents.get_image_summary(
    'C:/Users/jk/Desktop/Gate/_output/avec2014_train_201705060006', tags, iter_tag)

# transfer data to draw
info = tfevents.get_info_data(values, tags, iter_tag)

# ['train/lr', 'train/err_mae', 'train/err_mse', 'train/loss']
print(tags)

# draw data
tfevents_draw.draw(info, tags[3], iter_tag)
