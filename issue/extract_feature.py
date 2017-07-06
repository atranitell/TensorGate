# -*- coding: utf-8 -*-
""" extract feature
    updated: 2017/05/19
"""
import os
import math
import numpy as np

import tensorflow as tf

import gate
from gate.utils.logger import logger


def extract_feature(data_name, chkp_path, layer_name):
    """ test for regression net
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'test', chkp_path)

        # load data
        images, labels, filenames = dataset.loads()

        # show content - check network info
        # import tools.checkpoint as checkpoint
        # chkp_file_path = checkpoint.get_latest_chkp_path(chkp_path)
        # checkpoint.print_checkpoint_variables(chkp_file_path)

        # create network
        logits, end_points = gate.net.factory.get_network(
            dataset.hps, 'test', images, dataset.num_classes)

        if layer_name in end_points:
            logger.info('Finding the specified layer %s' % layer_name)
        else:
            logger.error('Could not find specified layer %s' % layer_name)
            return

        # get saver
        saver = tf.train.Saver(name='restore_feature')

        with tf.Session() as sess:
            # get latest checkpoint
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queuerunner.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))

            for cur in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break
                # running session to acuqire value
                _feat, _names = sess.run([end_points[layer_name], filenames])
                if cur == 0:
                    features = _feat
                    names = _names.tolist()
                else:
                    features = np.row_stack((features, _feat))
                    names += _names.tolist()

                if cur % 10 == 0:
                    logger.info('Has processed %d samples.' %
                                (cur * dataset.batch_size))

            logger.info('Has processed %d samples.' % dataset.total_num)
            logger.info('Saving the features to file...')
            features = features[0: dataset.total_num]
            names = np.array(names)[0: dataset.total_num]

            save_path = os.path.join(
                chkp_path, data_name + '_' + global_step + '_' + layer_name)
            np.save(save_path + '_features.npy', features)
            np.save(save_path + '_names.npy', names)

            logger.info('Feature shape is ' + str(features.shape))
            logger.info('All features has been saved in %s.' % save_path)

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
