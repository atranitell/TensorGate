
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf

import time
import math
from nets import nets_factory
from data import datasets_factory
from tensorflow.contrib import framework
from tensorflow.contrib import layers
import optimizer as opt_chooser


def test(name='cifar10', net_name='cifarnet', model_path=None):

    with tf.Graph().as_default() as g:
        # prepare the data
        dataset = datasets_factory.get_dataset(name, 'test')
        images, labels = dataset.loads()
        # ATTENTION!
        dataset.log.test_dir = model_path + '/test/'

        # acquire network
        logits, end_points = nets_factory.get_network(
            net_name, 'test', images, dataset.num_classes)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # load in the ckpt
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
            else:
                print('Non checkpoint file found in %s' % model_path)

            # start to queue runner
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            # prepare
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            aver_precision = 0
            aver_loss = 0
            tf.train.start_queue_runners(sess=sess)
            step = 0
            while step < num_iter and not coord.should_stop():
                aver_loss += sess.run(losses)
                predictions = tf.to_int32(tf.argmax(logits, axis=1))
                precision = sess.run(tf.reduce_mean(
                    tf.to_float(tf.equal(predictions, labels))))
                aver_precision += precision
                step += 1

            aver_precision = 1.0 * aver_precision / num_iter
            aver_loss = 1.0 * aver_loss / num_iter
            print('INFO:tensorflow:Iter in %d, precision: %f' %
                  (int(global_step), aver_precision))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def restore(chkp_path, saver, sess):
    ckpt = tf.train.get_checkpoint_state(chkp_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split(
            '/')[-1].split('-')[-1]
    else:
        print('Non checkpoint file found in %s' % chkp_path)


def train(data_name, net_name, chkp_path=None):
    # setting output level
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # get all param information
        dataset = datasets_factory.get_dataset(data_name, 'train')
        # reset_training_path
        if chkp_path is not None:
            dataset.log.train_dir = chkp_path

        # create global step
        with tf.device(dataset.device):
            global_step = framework.create_global_step()

        # get data
        images, labels = dataset.loads()

        # choose network
        fc_last, end_points = nets_factory.get_network(
            net_name, 'train', images, dataset.num_classes)

        # loss
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=labels, logits=logits)

        # regression
        # labels = tf.to_float(tf.div(labels, 100))
        labels = tf.to_float(labels)
        logits = layers.fully_connected(
            fc_last, 1,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.truncated_normal_initializer(
                stddev=1 / 192.0),
            weights_regularizer=None,
            activation_fn=None,
            scope='last_logits')
        end_points['last_logits'] = logits

        losses = tf.nn.l2_loss([labels-logits], name='l2_loss')

        # Gather initial summaries.
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        if dataset.lr.moving_average_decay:
            moving_average_variables = framework.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                dataset.lr.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # optimization
        with tf.device(dataset.device):
            learning_rate = opt_chooser.configure_learning_rate(
                dataset, dataset.total_num, global_step)
            optimizer = opt_chooser.configure_optimizer(dataset, learning_rate)
            # summaries.add(tf.summary.scalar('learning_rate', learning_rate,
            # name='learning_rate'))

        # compute gradients
        variables_to_train = tf.trainable_variables()
        grads = tf.gradients(losses, variables_to_train)
        train_op = optimizer.apply_gradients(
            zip(grads, variables_to_train),
            global_step=global_step, name='train_step')

        # precision
        # predictions = tf.to_int32(tf.argmax(logits, axis=1))

        # precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        loss = tf.reduce_mean(losses)

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': global_step,
                     'loss': loss},
                    #  'precision': precision},
            every_n_iter=dataset.log.print_frequency)

        chkp_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=dataset.log.train_dir,
            save_steps=dataset.log.save_model_iter,
            saver=tf.train.Saver(
                var_list=tf.global_variables(), max_to_keep=10),
            checkpoint_basename=dataset.name + '.ckpt')

        # running operation
        class running_hook(tf.train.SessionRunHook):

            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                    [global_step, loss],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                cur_iter = run_values.results[0]
                if (cur_iter - 1) % dataset.log.test_interval == 0 and (cur_iter - 1) != 0:
                    test(data_name, net_name, dataset.log.train_dir)

        with tf.train.MonitoredTrainingSession(
                hooks=[logging_hook, chkp_hook, running_hook()],
                save_summaries_steps=0,
                config=tf.ConfigProto(allow_soft_placement=False),
                checkpoint_dir=chkp_path) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def interface(args):
    """ interface related to command
    """
    data_name = 'cifar10'
    net_name = 'cifarnet'

    # check model
    if isinstance(args.model, str):
        if not tf.gfile.IsDirectory(args.model):
            raise ValueError('Error model path: ', args.model)

    # start to train
    if args.task == 'train' and args.model is None:
        train(data_name, net_name, chkp_path=None)

    # continue to train
    elif args.task == 'train' and args.model is not None:
        train(data_name, net_name, args.model)

    # test
    elif args.task == 'eval' and args.model is not None:
        test(data_name, net_name, model_path=args.model)

    # finetune

    # feature

    else:
        raise ValueError('Error task type ', args.task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='train', dest='task',
                        help='train/eval/finetune/feature')
    parser.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder: automatically use newest model')
    opt, arg = parser.parse_known_args()
    interface(opt)
