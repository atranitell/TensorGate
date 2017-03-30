
import tensorflow as tf
from gate import solver


class Updater():

    def __init__(self, dataset, global_step, losses, exclusions=None, is_summary=True):

        with tf.device(dataset.device):
            self.learning_rate = solver.updater_learning_rate.configure(
                dataset, dataset.total_num, global_step)
            self.optimizer = solver.updater_optimizer.configure(
                dataset, self.learning_rate)

        with tf.name_scope('updater'):
            tf.summary.scalar('lr', self.learning_rate)

        if exclusions is not None:
            variables_to_restore = []
            for var in tf.global_variables():
                excluded = False
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)
            self.saver = tf.train.Saver(var_list=variables_to_restore)
            self.variables_to_train = variables_to_restore
        else:
            self.saver = tf.train.Saver()
            self.variables_to_train = tf.trainable_variables()

        print('[NET] Variables will be trained as list:')
        for v in self.variables_to_train:
            print('[NET] ', v)

        # self.grads = tf.gradients(losses, self.variables_to_train)
        self.grads = self.optimizer.compute_gradients(losses)
        self.train_op = self.optimizer.apply_gradients(
            self.grads, global_step=global_step, name='train_step')

        # if dataset.lr.moving_average_decay:
        #     variable_averages = tf.train.ExponentialMovingAverage(
        #         dataset.lr.moving_average_decay, global_step)
        #     maintain_averages_op = variable_averages.apply(
        #         self.variables_to_train)
        # else:
        #     maintain_averages_op = None

        # with tf.control_dependencies([apply_grad_op, maintain_averages_op]):
        #     self.train_op = tf.no_op(name='train')

        # # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)

        # # Add histograms for gradients.
        # for grad, var in self.grads:
        #     if grad is not None:
        #         tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.name_scope('grads'):
            for grad, v in self.grads:
                prefix = v.op.name
                if prefix.find('global_step') == 0:
                    continue
                tf.summary.scalar(name=prefix + '_mean',
                                  tensor=tf.reduce_mean(grad))
                tf.summary.scalar(name=prefix + '_max',
                                  tensor=tf.reduce_max(grad))
                tf.summary.scalar(name=prefix + '_sum',
                                  tensor=tf.reduce_sum(grad))

        with tf.name_scope('weights'):
            for v in tf.trainable_variables():
                prefix = v.op.name
                if prefix.find('global_step') == 0:
                    continue
                tf.summary.scalar(name=prefix + '_mean',
                                  tensor=tf.reduce_mean(v))
                tf.summary.scalar(name=prefix + '_max',
                                  tensor=tf.reduce_max(v))
                tf.summary.scalar(name=prefix + '_sum',
                                  tensor=tf.reduce_sum(v))

    def get_variables_to_train(self):
        return self.variables_to_train

    def get_variables_saver(self):
        return self.saver

    def get_learning_rate(self):
        return self.learning_rate

    def get_train_op(self):
        return self.train_op
