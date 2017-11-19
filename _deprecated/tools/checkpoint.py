
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def get_latest_chkp_path(chkp_fold_path):
    ckpt = tf.train.get_checkpoint_state(chkp_fold_path)
    return ckpt.model_checkpoint_path


def write_checkpoint_model_items(chkp_file_path, chkp_model_list):
    """ Write model items to checkpoint file
        if chkp_file_path has existed, it will raise a error.
    CHKP_MODEL_LIST is a list:
        ['train.ckpt-3001', 'train.ckpt-1001', 'train.ckpt-2001', 'train.ckpt-3001']
    CHKP_FILE will be written: chkp_model_list[0] will as a index.
        model_checkpoint_path: "train.ckpt-3001"
        all_model_checkpoint_paths: "train.ckpt-1001"
        all_model_checkpoint_paths: "train.ckpt-2001"
        all_model_checkpoint_paths: "train.ckpt-3001"
    """
    if not os.path.exists(chkp_file_path):
        raise ValueError('checkpoint file path is wrong.')
    with open(chkp_file_path, 'w') as fp:
        fp.write('model_checkpoint_path: "' + chkp_model_list[0] + '"\n')
        for idx in range(1, len(chkp_model_list)):
            fp.write('all_model_checkpoint_paths: "' +
                     chkp_model_list[idx] + '"\n')


def get_checkpoint_model_items(chkp_file_path):
    """ Extract checkpoint model items to list
    e.g.
        model_list[0]: model_checkpoint_path: "train.ckpt-3001"
        model_list[1]: all_model_checkpoint_paths: "train.ckpt-1001"
        model_list[2]: all_model_checkpoint_paths: "train.ckpt-2001"
        model_list[3]: all_model_checkpoint_paths: "train.ckpt-3001"
    """
    chkp_model_list = []
    with open(chkp_file_path) as fp:
        for line in fp:
            r = re.findall('\"(.*?)\"', line)
            if len(r):
                chkp_model_list.append(r[0])
    return chkp_model_list


def print_checkpoint_variables(filepath):
    """ Print variables name in checkpoint file.
    e.g.
        #  print_checkpoint_variables('train.ckpt-100001') 
    """
    reader = pywrap_tensorflow.NewCheckpointReader(filepath)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print("tensor_size: ", var_to_shape_map[key])


def print_checkpoint_specific_tensor(filepath, name):
    reader = pywrap_tensorflow.NewCheckpointReader(filepath)
    if reader.has_tensor(name):
        print("tensor_name: ", name)
        print(reader.get_tensor(name))


def change_checkpoint_prefix(infile, outfile, old_prefix, new_prefix):
    """ Solve the issue of different name scope of same model.
    e.g.
        # change_checkpoint_prefix(
        #     infile='vgg_16.ckpt', outfile='vgg_16_new',
        #     old_prefix='vgg_16', new_prefix='net/vgg_16')
    """
    with tf.Graph().as_default():
        reader = pywrap_tensorflow.NewCheckpointReader(infile)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # save to Graph
        for key in var_to_shape_map:
            print("tensor_name_old: ", key)
            new_key = key.replace(old_prefix, new_prefix)
            tf.Variable(reader.get_tensor(key), name=new_key)
            print("tensor_name_new: ", new_key)
        # save to file
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, outfile)


def add_checkpoint_prefix(infile, outfile, new_prefix):
    """ Solve the issue of different name scope of same model.
    e.g.
        # change_checkpoint_prefix(
        #     infile='vgg_16.ckpt', outfile='vgg_16_new',
        #     old_prefix='vgg_16', new_prefix='net/vgg_16')
    """
    with tf.Graph().as_default():
        reader = pywrap_tensorflow.NewCheckpointReader(infile)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # save to Graph
        for key in var_to_shape_map:
            tf.Variable(reader.get_tensor(key), name=new_prefix + '/' + key)
            print("tensor_name_new: ", new_prefix + '/' + key)
        # save to file
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, outfile)


def merge_checkpoint(infile_src, infile_dst, outfile):
    """ Solve the model using different optimizer,
        copy old tensor to new model.
    e.g.
        # merge_checkpoint(
        #    'train.ckpt-adam',
        #    'train.ckpt-momentum',
        #    'train.ckpt-merge')

        'train.ckpt-merge' will use momentum optimizer but all weights
            and bias derived from 'train.ckpt-adam'.
    """
    with tf.Graph().as_default():
        reader_src = pywrap_tensorflow.NewCheckpointReader(infile_src)
        var_to_shape_map_src = reader_src.get_variable_to_shape_map()
        reader_dst = pywrap_tensorflow.NewCheckpointReader(infile_dst)
        var_to_shape_map_dst = reader_dst.get_variable_to_shape_map()
        for key in var_to_shape_map_dst:
            if key in var_to_shape_map_src:
                tf.Variable(reader_src.get_tensor(key), name=key)
                print("tensor changed from: ", key)
            else:
                tf.Variable(reader_dst.get_tensor(key), name=key)
                print("tensor unchanged from: ", key)

        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, outfile)


def merge_fuse_checkpoint(model_1, model_2, fuse_model, outfile):
    """
    e.g.
    #    merge_fuse_checkpoint(
    #        model_1='train.ckpt-100001',
    #        model_2='flow_train.ckpt-100001',
    #        fuse_model='fuse_train.ckpt-1',
    #        outfile='fuse_train.ckpt-merge')
    """
    print(' Changing the prefix of model1')
    change_checkpoint_prefix(model_1, model_1 + '_1', 'net', 'net1')
    print('\n Changing the prefix of model2')
    change_checkpoint_prefix(model_2, model_2 + '_2', 'net', 'net2')
    print('\n Merge Model1')
    merge_checkpoint(model_1 + '_1', fuse_model, fuse_model + '_tmp')
    print('\n Merge Model2')
    merge_checkpoint(model_2 + '_2', fuse_model + '_tmp', outfile)
    print('\n Finished!')


def compute_parameter_num(filepath):
    """ compute sum weights in DNN model
    e.g.
        compute_parameter_num('train.ckpt-100001')
    """
    def _inner_product(arr):
        _val = 1
        for _i in arr:
            _val *= _i
        return _val

    reader = pywrap_tensorflow.NewCheckpointReader(filepath)
    var_to_shape_map = reader.get_variable_to_shape_map()
    val_list = []
    for key in var_to_shape_map:
        if key.find('Adam') > 0:
            continue
        print("tensor_name: ", key)
        val = _inner_product(var_to_shape_map[key])
        print("tensor_size: ", var_to_shape_map[key], val)
        val_list.append(val)
    print('Total params number: ', sum(val_list))
    return sum(val_list)
