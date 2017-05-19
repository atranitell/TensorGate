# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from tensorflow.contrib import framework
import gate


class Regression():

    def __init__(self):
        """ a choose for different regression
        """
        pass

    def inference(self, dataset, phase, exclusions=None):
        """
        """
        # -------------------------------------------
        # build data model
        # -------------------------------------------
        images, labels, _ = dataset.loads()

        # -------------------------------------------
        # get global step
        # -------------------------------------------
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # -------------------------------------------
        # get Deep Neural Network
        # for regression issue, num_class should be 1
        # -------------------------------------------
        logits, end_points = gate.net.factory.get_network(
            net_name, phase, images, 1,
            dataset.hps, name_scope='')

        # -------------------------------------------
        # get loss
        # -------------------------------------------
        losses, labels, logits = gate.loss.l2.get_loss(
            logits, labels, dataset.num_classes, dataset.batch_size)

        # -------------------------------------------
        # get error
        # -------------------------------------------
        mae, rmse = gate.loss.l2.get_error(
            logits, labels, dataset.num_classes)

        running_hook = self.Running_Hook(
            global_step, losses, mae, rmse, dataset)

        return running_hook

    class Running_Hook(tf.train.SessionRunHook):

        def __init__(self, global_step, losses, mae, rmse, dataset):
            self.global_step = global_step
            self.losses = losses
            self.mae = mae
            self.rmse = rmse
            self.learning_rate = -1

            self.duration, self.start_time = 0, 0
            self.mean_loss, self.mean_mae, self.mean_rmse = 0, 0, 0
            self.best_iter_mae, self.best_mae = 0.0, 100000.0
            self.best_iter_rmse, self.best_rmse = 0.0, 100000.0

        def set_learning_rate(self, learning_rate):
            self.learning_rate = learning_rate

        def before_run(self, run_context):
            self.start_time = time.time()
            run_args = [self.global_step, self.losses,
                        self.mae, self.rmse, self.learning_rate]
            return tf.train.SessionRunArgs(run_args, feed_dict=None)

        def after_run(self, run_context, run_values):
            # accumulate datas
            cur_iter = run_values.results[0] - 1
            self.mean_loss += run_values.results[1]
            self.mean_mae += run_values.results[2]
            self.mean_rmse += run_values.results[3]
            self.duration += (time.time() - self.start_time)

            # print information
            if cur_iter % dataset.log.print_frequency == 0:
                _invl = dataset.log.print_frequency
                _loss = self.mean_loss / _invl
                _mae = self.mean_mae / _invl
                _rmse = math.sqrt(self.mean_mse / _invl)
                _lr = str(run_values.results[4])
                _duration = self.duration * 1000 / _invl
                # there time is the running time of a iteration
                # (if 1 GPU, it is a batch)
                format_str = 'Iter:%d, loss:%.4f, mae:%.4f, rmse:%.4f, lr:%s, time:%.2fms.'
                gate.utils.show.TRAIN(
                    format_str % (cur_iter, _loss, _mae, _rmse, _lr, _duration))
                # set zero
                self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0

            # evaluation
            if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                test_start_time = time.time()
                test_mae, test_rmse = test(data_name, net_name,
                                           dataset.log.train_dir, summary_test)
                test_duration = time.time() - test_start_time

                if test_mae < self.best_mae:
                    self.best_mae = test_mae
                    self.best_iter_mae = cur_iter
                if test_rmse < self.best_rmse:
                    self.best_rmse = test_rmse
                    self.best_iter_rmse = cur_iter

                gate.utils.show.TEST(
                    'Test Time: %fs, best MAE: %f in %d, best RMSE: %f in %d.' %
                    (test_duration, self.best_mae, self.best_iter_mae,
                        self.best_rmse, self.best_iter_rmse))
