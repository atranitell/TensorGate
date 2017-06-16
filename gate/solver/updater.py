
import tensorflow as tf
from gate import solver
from gate import utils
from gate.utils.logger import logger


class Updater():
    """ Updater
    """

    def __init__(self):
        self.learning_rate = None
        self.optimizer = None
        self.grads = None
        self.saver = None
        self.train_op = None
        self.variables_to_train = None
        self.variables_to_restore = None

    def get_learning_rate(self, dataset=None, global_step=None):
        if self.learning_rate is not None:
            return self.learning_rate
        utils.check.raise_none_param(dataset, global_step)
        return solver.updater_learning_rate.configure(
            dataset, dataset.total_num, global_step)

    def get_optimizer(self, dataset=None):
        if self.optimizer is not None:
            return self.optimizer
        utils.check.raise_none_param(dataset, self.learning_rate)
        return solver.updater_optimizer.configure(
            dataset, self.learning_rate)

    def get_gradients(self, losses=None, opt=None):
        if self.grads is not None:
            return self.grads
        utils.check.raise_none_param(losses, self.optimizer)
        self.grads = self.optimizer.compute_gradients(
            losses, var_list=self.variables_to_train)

        # clip
        if opt.clip_method is 'clip_by_value':
            cmin = opt.clip_value_min
            cmax = opt.clip_value_max
            self.grads = [(tf.clip_by_value(grad, cmin, cmax), var)
                          for grad, var in self.grads]

        return self.grads

    def get_train_op(self):
        if self.train_op is not None:
            return self.train_op
        else:
            raise ValueError('train op does not exist.')

    def _inclusion_var(self, exclusions, var_list):
        """ exclude prefix elements of exclusions in var_list
        """
        _vars = []
        for var in var_list:
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                _vars.append(var)
        return _vars

    def get_trainable_list(self, exclusions=None):
        """ var for train.
        """
        if self.variables_to_train is not None:
            return self.variables_to_train
        if exclusions is not None:
            return self._inclusion_var(exclusions, tf.trainable_variables())
        else:
            return tf.trainable_variables()

    def get_restore_list(self, exclusions=None):
        """ import variables excluded from exclusions.
        """
        if self.variables_to_restore is not None:
            return self.variables_to_restore
        if exclusions is not None:
            return self._inclusion_var(exclusions, tf.global_variables())
        else:
            return tf.global_variables()

    def get_variables_saver(self):
        """ restore list.
        """
        if self.saver is not None:
            return self.saver
        utils.check.raise_none_param(self.variables_to_restore)
        return tf.train.Saver(var_list=self.variables_to_restore,
                              name='restore', allow_empty=True)

    def init_default_updater(self, dataset, global_step,
                             losses, exclusions=None):
        """ init_default_updater
        """
        # choose part of weights to restore or train
        if exclusions is not None and 'restore' in exclusions:
            ex_restore = exclusions['restore']
        else:
            ex_restore = None
        if exclusions is not None and 'train' in exclusions:
            ex_train = exclusions['train']
        else:
            ex_train = None

        self.learning_rate = self.get_learning_rate(dataset, global_step)
        self.optimizer = self.get_optimizer(dataset)

        # variables_to_train just includes items in exclusions
        self.variables_to_train = self.get_trainable_list(ex_train)

        # normal compute gradients
        self.grads = self.get_gradients(losses, dataset.opt)
        grad_op = self.optimizer.apply_gradients(
            self.grads, global_step=global_step, name='train_step')

        # add moving average decay
        if dataset.lr.moving_average_decay is not None:
            variable_averages = tf.train.ExponentialMovingAverage(
                dataset.lr.moving_average_decay, global_step)
            avg_op = variable_averages.apply(self.variables_to_train)
            with tf.control_dependencies([grad_op, avg_op]):
                self.train_op = tf.no_op(name='train')
        else:
            self.train_op = grad_op

        # restore weights from self.variables to restore
        self.variables_to_restore = self.get_restore_list(ex_restore)
        self.saver = self.get_variables_saver()

        # print info
        self.print_restore_list()
        self.print_trainable_list()
        self.print_grads_list()
        self.print_global_list()

        # summary related
        self._summary_lr()
        # self._summary_grad(True, True)
        # self._summary_weight()

    # def init_layerwise_updater(self, dataset, global_step,
    #                            losses, prefix, coeff, exclusions=None):
    #     """ The updater method will adjust learning rate
    #             for every variables in according to different lr.
    #     """
    #     # acquire trainable list
    #     self.variables_to_train, self.variables_to_restore = self.get_trainable_list(exclusions)
    #     self.saver = self.get_variables_saver()

    #     # setting layerwise coff
    #     lr_coeff = {}
    #     for weight in self.variables_to_train:
    #         if weight.op.name.find(prefix) >= 0:
    #             lr_coeff[weight.op.name] = coeff

    #     self.learning_rate = self.get_learning_rate(dataset, global_step)
    #     self.optimizer = self.get_optimizer(dataset)

    #     gradients = self.optimizer.compute_gradients(losses)
    #     # adjust grads according to layerwise
    #     self.grads = []
    #     for grad, var in gradients:
    #         if grad is None:
    #             continue
    #         if var.op.name in lr_coeff:
    #             logger.net(str(lr_coeff[var.op.name]) + ' ' + var.op.name)
    #             grad *= lr_coeff[var.op.name]
    #         self.grads.append((grad, var))

    #     # start to train
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         self.train_op = self.get_train_op(global_step)

    #     # print info
    #     self.print_trainable_list()
    #     # self.print_grads_list()

    #     # add to summary
    #     self._summary_lr()
    #     # self._summary_grad()
    #     # self._summary_weight()

    def _summary_grad(self, grad_summary=True, grad_hist=False):
        """ input:
                self.grad
        """
        with tf.name_scope('grads'):
            for grad, var in self.grads:
                prefix = var.op.name
                if prefix.find('global_step') == 0 or grad is None:
                    continue
                if grad_summary:
                    tf.summary.scalar(var.op.name + '_mean',
                                      tf.reduce_mean(grad))
                    tf.summary.scalar(var.op.name + '_max',
                                      tf.reduce_max(grad))
                    tf.summary.scalar(var.op.name + '_sum',
                                      tf.reduce_sum(grad))
                if grad_hist:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

    def _summary_weight(self, weight_summary=True, weight_hist=False):
        with tf.name_scope('weights'):
            for weight in tf.trainable_variables():
                prefix = weight.op.name
                if prefix.find('global_step') == 0 or weight is None:
                    continue
                if weight_summary:
                    tf.summary.scalar(weight.op.name + '_mean',
                                      tf.reduce_mean(weight))
                    tf.summary.scalar(weight.op.name + '_max',
                                      tf.reduce_max(weight))
                    tf.summary.scalar(weight.op.name + '_sum',
                                      tf.reduce_sum(weight))
                # Add histograms for trainable variables.
                if weight_hist:
                    tf.summary.histogram(weight.op.name, weight)

    def _summary_lr(self, lr_summary=True):
        tf.summary.scalar('train/lr', self.learning_rate)

    def print_trainable_list(self):
        logger.sys('TRAINABLE LIST:')
        for var in self.variables_to_train:
            logger.sys(str(var))

    def print_global_list(self):
        logger.net('ALL VARIABLES:')
        for var in tf.global_variables():
            logger.net(str(var))

    def print_grads_list(self):
        logger.sys('Gradients will be trained as list:')
        for grad, var in self.grads:
            logger.sys(str(grad))

    def print_restore_list(self):
        logger.sys('RESTORE LIST:')
        for var in self.variables_to_restore:
            logger.sys(str(var))
