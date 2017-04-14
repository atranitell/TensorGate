
import tensorflow as tf
from gate.utils import show


class Snapshot():

    def __init__(self):
        self.chkp_hook = None
        self.summary_hook = None
        self.summary_test = None

    def get_chkp_hook(self, dataset):
        if self.chkp_hook is None:
            self.chkp_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=dataset.log.train_dir,
                save_steps=dataset.log.save_model_iter,
                saver=tf.train.Saver(var_list=tf.global_variables(),
                                     max_to_keep=10000, name='save_all'),
                checkpoint_basename=dataset.name + '.ckpt')
        return self.chkp_hook

    def get_summary_hook(self, dataset):
        if self.summary_hook is None:
            self.summary_hook = tf.train.SummarySaverHook(
                save_steps=dataset.log.save_summaries_iter,
                output_dir=dataset.log.train_dir,
                summary_op=tf.summary.merge_all())
        return self.summary_hook

    def get_summary_test(self, dataset):
        if self.summary_test is None:
            self.summary_test = tf.summary.FileWriter(dataset.log.train_dir)
        return self.summary_test

    def restore(self, sess, chkp_fold, saver):
        ckpt = tf.train.get_checkpoint_state(chkp_fold)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            show.SYS('Load checkpoint from: %s' % ckpt.model_checkpoint_path)
        else:
            show.TEST('Non checkpoint file found in %s' % ckpt.model_checkpoint_path)
        return global_step
