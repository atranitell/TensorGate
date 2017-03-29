
import tensorflow as tf
from gate import solver


class Updater():

    def __init__(self, dataset, global_step, losses, exclusions=None, is_summary=True):

        with tf.device(dataset.device):
            self.learning_rate = solver.updater_learning_rate.configure(
                dataset, dataset.total_num, global_step)
            self.optimizer = solver.updater_optimizer.configure(dataset, self.learning_rate)
        
        with tf.name_scope('train'):
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

        self.grads = tf.gradients(losses, self.variables_to_train)
        self.train_op = self.optimizer.apply_gradients(
            zip(self.grads, self.variables_to_train),
            global_step=global_step, name='train_step')

        if is_summary:
            with tf.name_scope('grads'):
                for idx, v in enumerate(self.grads):
                    prefix = self.variables_to_train[idx].name
                    if prefix.find('global_step') == 0:
                        continue
                    tf.summary.scalar(name=prefix + '_mean', tensor=tf.reduce_mean(v))
                    tf.summary.scalar(name=prefix + '_max', tensor=tf.reduce_max(v))
                    tf.summary.scalar(name=prefix + '_sum', tensor=tf.reduce_sum(v))

    def get_variables_to_train(self):
        return self.variables_to_train

    def get_variables_saver(self):
        return self.saver

    def get_learning_rate(self):
        return self.learning_rate

    def get_train_op(self):
        return self.train_op
