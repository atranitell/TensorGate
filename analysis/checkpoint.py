
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def print_checkpoint_variables(filepath):
    """ Print variables name in checkpoint file.
    e.g.
        #  print_checkpoint_variables('train.ckpt-100001') 
    """
    reader = pywrap_tensorflow.NewCheckpointReader(filepath)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


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
            tf.Variable(reader.get_tensor(key), name=new_prefix+'/'+key)
            print("tensor_name_new: ", new_prefix+'/'+key)
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

# merge_fuse_checkpoint(
#     model_1='C:/Users/jk/Desktop/Gate/_output/t1/avec2014_train.ckpt-100001',
#     model_2='C:/Users/jk/Desktop/Gate/_output/t1/avec2014_flow_train.ckpt-43401',
#     fuse_model='C:/Users/jk/Desktop/Gate/_output/t1/avec2014_fuse_16f_succ_train.ckpt-1',
#     outfile='C:/Users/jk/Desktop/Gate/_output/t1/avec2014_fuse_16f_succ_train.ckpt-merge'
# )

print_checkpoint_variables('C:/Users/jk/Desktop/Gate/_output/vgg/vggface16.tfmodel')

# add_checkpoint_prefix(infile='C:/Users/jk/Desktop/Gate/_output/facenet_webface/model-20170511-185253.ckpt-80000',
#                       outfile='C:/Users/jk/Desktop/Gate/_output/facenet_webface/inception-resnet-v1',
#                       new_prefix='net')
