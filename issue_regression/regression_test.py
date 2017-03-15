
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from nets import nets_factory
from data import datasets_factory

from util_tools import output


def test(name, net_name, model_path=None):

    with tf.Graph().as_default() as g:
        #-------------------------------------------
        # Preparing the dataset
        #-------------------------------------------
        dataset = datasets_factory.get_dataset(name, 'test')
        dataset.log.test_dir = model_path + '/test/'

        output.print_basic_information(dataset)

        images, labels_orig = dataset.loads()

        #-------------------------------------------
        # Network
        #-------------------------------------------
        logits, end_points = nets_factory.get_network(
            net_name, 'test', images, 1)

        # ATTENTION!
        labels = tf.to_float(tf.reshape(labels_orig, [dataset.batch_size, 1]))
        labels = tf.div(labels, dataset.num_classes)
        losses = tf.nn.l2_loss([labels - logits], name='l2_loss')

        err_mae = tf.reduce_mean(
            input_tensor=tf.abs((logits - labels) * dataset.num_classes), name='err_mae')
        err_mse = tf.reduce_mean(
            input_tensor=tf.square((logits - labels) * dataset.num_classes), name='err_mse')


        saver = tf.train.Saver()
        with tf.Session() as sess:
            #-------------------------------------------
            # restore from checkpoint 
            #-------------------------------------------
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
                print('[TEST] Load checkpoint from: %s' %
                      ckpt.model_checkpoint_path)
            else:
                print('[TEST] Non checkpoint file found in %s' %
                      ckpt.model_checkpoint_path)

            #-------------------------------------------
            # start queue from runner
            #-------------------------------------------
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            
            #-------------------------------------------
            # evaluation
            #-------------------------------------------
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            mae = 0.0
            rmse = 0.0
            loss = 0.0

            for _ in range(num_iter):
                if coord.should_stop():
                    break
                loss += sess.run(losses)
                mae += sess.run(err_mae)
                rmse += sess.run(err_mse)

            loss = 1.0 * loss / num_iter
            rmse = math.sqrt(1.0 * rmse / num_iter)
            mae = 1.0 * mae / num_iter

            #-------------------------------------------
            # output
            #-------------------------------------------
            print('[TEST] Iter:%d, total test sample:%d, num_batch:%d' %
                  (int(global_step), dataset.total_num, num_iter))

            print('[TEST] loss:%.2f, mae:%.2f, rmse:%.2f' %
                  (loss, mae, rmse))

            #-------------------------------------------
            # terminate all threads
            #-------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
