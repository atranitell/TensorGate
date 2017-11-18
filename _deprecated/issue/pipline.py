# -*- coding: utf-8 -*-
""" Processing multi-model for specific task.
    updated: 2017/07/06
"""
import os
import gate
import tools


def pipline(name, chkp_path, fn, **fn_args):
    """ test all model in checkpoint file
    """
    chkp_file_path = os.path.join(chkp_path, 'checkpoint')
    # make a backup
    gate.utils.filesystem.copy_file(chkp_file_path, chkp_file_path + '.bk')
    # acquire model list
    chkp_model_list = tools.checkpoint.get_checkpoint_model_items(
        chkp_file_path)

    for idx in range(1, len(chkp_model_list)):
        # write list to new checkpoint file
        new_model_list = chkp_model_list.copy()
        new_model_list[0] = new_model_list[idx]
        tools.checkpoint.write_checkpoint_model_items(
            chkp_file_path, new_model_list)
        gate.utils.logger.logger.info('Process model %s' % new_model_list[0])
        ''' run fn
        '''
        fn(name, chkp_path, **fn_args)
