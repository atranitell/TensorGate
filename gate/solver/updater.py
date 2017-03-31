
import tensorflow as tf
from gate import solver


class Updater():

    def __init__(self, method, weight_summary=True, grad_summary=True,
                 weight_hist=False, grad_hist=False, **kwarg):

        if method == 'fuse':
            pass
        else:
            self._init_default_updater(kwarg)

        # summary
        self._summary_gard(grad_summary, grad_hist)
        self._summary_weight(weight_summary, weight_hist)

    def _init_default_updater(self, dataset=None, global_step=None,
                              losses=None, exclusions=None):
        """
        """
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

    def _init_layerwise_updater(self, dataset=None, global_step=None,
                                losses=None, exclusions=None):
        """ The updater method will adjust learning rate
                for every variables in according to different lr.
        """
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

    def _moving_average_decay(self):
        """
        """
        # if dataset.lr.moving_average_decay:
        #     variable_averages = tf.train.ExponentialMovingAverage(
        #         dataset.lr.moving_average_decay, global_step)
        #     maintain_averages_op = variable_averages.apply(
        #         self.variables_to_train)
        # else:
        #     maintain_averages_op = None

        # with tf.control_dependencies([apply_grad_op, maintain_averages_op]):
        #     self.train_op = tf.no_op(name='train')
        raise ValueError('This function has not realized!')

    def _summary_gard(self, grad_summary, grad_hist):
        """ input:
                self.grad
        """
        with tf.name_scope('grads'):
            if grad_summary:
                for grad, v in self.grads:
                    prefix = v.op.name
                    if prefix.find('global_step') == 0:
                        continue
                    tf.summary.scalar(name=prefix + '_mean', tensor=tf.reduce_mean(grad))
                    tf.summary.scalar(name=prefix + '_max', tensor=tf.reduce_max(grad))
                    tf.summary.scalar(name=prefix + '_sum', tensor=tf.reduce_sum(grad))
            # Add histograms for gradients.
            if grad_hist:
                for grad, var in self.grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)

    def _summary_weight(self, weight_summary, weight_hist):
        with tf.name_scope('weights'):
            if weight_summary:
                for weight in tf.trainable_variables():
                    prefix = weight.op.name
                    if prefix.find('global_step') == 0:
                        continue
                    tf.summary.scalar(name=prefix + '_mean', tensor=tf.reduce_mean(weight))
                    tf.summary.scalar(name=prefix + '_max', tensor=tf.reduce_max(weight))
                    tf.summary.scalar(name=prefix + '_sum', tensor=tf.reduce_sum(weight))
            # Add histograms for trainable variables.
            if weight_hist:
                for weight in tf.trainable_variables():
                    tf.summary.histogram(weight.op.name, weight)

    def get_variables_to_train(self):
        return self.variables_to_train

    def get_variables_saver(self):
        return self.saver

    def get_learning_rate(self):
        return self.learning_rate

    def get_train_op(self):
        return self.train_op
