# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

Checkpoint helper

In tensorflow, there are two method to save a model. In general, we use the 
checkpoint file, it save model including:
- model.data-000000-of-000001
- model.index
- model.meta

In addition, we could freeze model parameters and only keep the input and
output. In the case, the model could be distributed to users.
- model.pb

Then, the input just meets the shape and type. The model could output the
output we expect.

related-checkpoint
- inspect the variables in checkpoint/pb
- load the variables in checkpoint/pb
- load the model of checkpoint/pb to network
- load a subset of checkpoint to network

e.g.
# 1. load pre-trained model and save model and graph file
with tf.Graph().as_default() as graph:
  .... # build your network
  # load pre-trained network
  var_list_op, var_list = checkpoint.ckpt_read('resnet_v2_50.ckpt')
  saver = tf.train.Saver(var_list=tf.global_variables())
  with tf.Session() as sess:
    sess.run(var_list_op, var_list)
    # save model weight
    saver.save(sess, 'resnet')
    # save pb graph file
    pb_write('resnet_graph.pb', graph)

# 2. freeze pb and weight to single *.pb file
ckpt_to_freezed_pb('resnet_graph.pb', 'resnet', 
                   'resnet_v2_50.pb', 'flatten/Reshape')

# 3. optimize the pb model
pb_optimize(
    input_graph='resnet_v2_50.pb',
    output_graph='resnet_v2_50_new.pb',
    input_names='DecodeJpeg/contents',
    output_names='flatten/Reshape'
)

# 4. reload new single model
graph = checkpoint.pb_read('resnet_v2_50_new.pb')
raw_data = tf.gfile.FastGFile('demo.JPEG', 'rb').read()
x = graph.get_tensor_by_name('prefix/DecodeJpeg/contents:0')
y = graph.get_tensor_by_name('prefix/flatten/Reshape:0')
y = tf.nn.top_k(y, 5)
with tf.Session(graph=graph) as sess:
  print(sess.run(y, feed_dict={x: raw_data}))

"""

import os
import re
import tensorflow as tf
from tensorflow.contrib import framework
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference
from tensorflow.python.tools import import_pb_to_tensorboard


def ckpt_read(path, ignore_miss_vars=True):
  """ from pre-trained model to distill suitable variables to import
    new network. only the exact same variable key (name) could be imported.

    The function allow us to import vars, a subset of customed network,
    from a pre-trained model.

  Return:
    import_vars_op: a op should be runed.
    import_vars: a feed-dict value.

  e.g.
    sess.run(import_vars_op, import_vars)
  """
  if os.path.isfile(path):
    return ckpt_read_from_file(path, ignore_miss_vars)
  elif os.path.isdir(path):
    return ckpt_read_from_dir(path, ignore_miss_vars)
  else:
    raise ValueError('Path %s is not existed.' % path)


def ckpt_read_from_file(filepath, ignore_miss_vars=True):
  """ load ckpt from file.
  e.g.
    filepath: resnet_v2_50/resnet_v2_50.ckpt/train.ckpt-3001
  """
  import_vars_op, import_vars = framework.assign_from_checkpoint(
      model_path=filepath,
      var_list=tf.global_variables(),
      ignore_missing_vars=ignore_miss_vars)
  # for var in import_vars:
  #   print('Imported: '+str(var))
  return import_vars_op, import_vars


def ckpt_read_from_dir(dirpath, ignore_miss_vars=True):
  """ load ckpt from checkpoint entries.
  """
  entry = ckpt_record_entry(dirpath)
  entry_path = os.path.join(dirpath, entry)
  return ckpt_read_from_file(entry_path, ignore_miss_vars)


def ckpt_record_entry(dirpath):
  """ If the path is a directory, it will automatically search for the file
    named 'checkpoint', else it will pinpoint to a specific filepath.

  Return: <list>[N]

  e.g.
  Input file content:
    model_checkpoint_path: "mnist.ckpt-503"
    all_model_checkpoint_paths: "mnist.ckpt-1"
    all_model_checkpoint_paths: "mnist.ckpt-501"

  Return content:
    checkpoint_list: 
      list<['mnist.ckpt-1', 'mnist.ckpt-501']>
  """
  # pinpoint checkpoint filepath
  if not os.path.isdir(dirpath):
    raise ValueError('The path is not a directory.')
  entry_path = os.path.join(dirpath, 'checkpoint')
  # index to list
  entry_list = []
  with open(entry_path) as fp:
    for line in fp:
      r = re.findall('\"(.*?)\"', line)
      if len(r) and line.find('all') == 0:
        entry_list.append(r[0])
  return entry_list


def ckpt_inspect(filepath):
  """ load ckpt from file
  e.g.
    filepath: resnet_v2_50/resnet_v2_50.ckpt/train.ckpt-3001
  Returns:
    a dict<key>{'key', 'shape', 'type', 'value'}
  """
  ret = {}
  reader = framework.load_checkpoint(filepath)
  var_to_dtype_map = reader.get_variable_to_dtype_map()
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    ret[key] = {
        'key': key,
        'shape': var_to_shape_map[key],
        'type': var_to_dtype_map[key],
        'value': reader.get_tensor(key)
    }
  return ret


def ckpt_inspect_print(filepath):
  """ print contents of ckpt
  """
  ret = ckpt_inspect(filepath)
  for key in ret:
    r = ret[key]
    print('%s, %s, %s' % (r['key'], r['shape'], r['type']))


def ckpt_to_freezed_pb(input_graph, input_ckpt, output_graph,
                       output_node_name, input_binary=True):
  """ It's useful to do this when we need to load a single file in C++, 
  especially in environments like mobile or embedded where we may not have 
  access to the RestoreTensor ops and file loading calls that they rely on.

  e.g. 
    input_graph=some_graph_def.pb
    input_checkpoint=model.ckpt-8361242
    output_graph=/tmp/frozen_graph.pb 
    output_node_names=softmax
    input_binary=True

  """
  from tensorflow.python.saved_model import tag_constants
  from tensorflow.core.protobuf import saver_pb2
  freeze_graph.freeze_graph(
      input_graph=input_graph,
      input_saver=None,
      input_binary=input_binary,
      input_checkpoint=input_ckpt,
      output_node_names=output_node_name,
      restore_op_name='',
      filename_tensor_name='',
      output_graph=output_graph,
      clear_devices=True,
      initializer_nodes='',
      variable_names_whitelist='',
      variable_names_blacklist='',
      input_meta_graph=None,
      input_saved_model_dir=None,
      saved_model_tags=tag_constants.SERVING,
      checkpoint_version=saver_pb2.SaverDef.V2)


def pb_read(filepath):
  """ load pb file content. We load the protobuf file from the disk and parse 
  it to retrieve the unserialized graph_def. Then, we import the graph_def 
  into a new Graph and returns it The name var will prefix every op/nodes in 
  your graph since we load everything in a new graph, this is not needed.
  """
  with tf.gfile.GFile(filepath, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name="prefix")
  return graph


def pb_write(filepath, graph):
  """ freeze graph to protobol binary file.
  filepath: dst path to save.
  graph: tf.Graph().as_default() (e.g.)
  """
  graph_def = graph.as_graph_def()
  with tf.gfile.GFile(filepath, 'wb') as f:
    f.write(graph_def.SerializeToString())


def pb_inspect_print(graph_path):
  for node in pb_read(graph_path).get_operations():
    print(node.name)


def pb_optimize(input_graph, output_graph,
                input_names, output_names,
                frozen_graph=True):
  """ Removes parts of a graph that are only needed for training.

  There are several common transformations that can be applied to GraphDefs
  created to train a model, that help reduce the amount of computation needed 
  when the network is used only for inference. These include:

  - Removing training-only operations like checkpoint saving.
  - Stripping out parts of the graph that are never reached.
  - Removing debug operations like CheckNumerics.
  - Folding batch normalization ops into the pre-calculated weights.
  - Fusing common operations into unified versions.

  This script takes a frozen GraphDef file (where the weight variables have been
  converted into constants by the freeze_graph script) and outputs a new GraphDef
  with the optimizations applied.

  Returns:
    An optimized version of the input graph.

  An example of command-line usage is:

  input_graph=some_graph_def.pb
  output_graph=/tmp/optimized_graph.pb
  input_names=Mul
  output_names=softmax
  frozen_graph: If true, the input graph is a binary frozen GraphDef
      file; if false, it is a text GraphDef proto file.  

  """
  from google.protobuf import text_format
  from tensorflow.core.framework import graph_pb2
  from tensorflow.python.framework import dtypes
  from tensorflow.python.framework import graph_io
  from tensorflow.python.platform import app
  from tensorflow.python.platform import gfile
  from tensorflow.python.tools import optimize_for_inference_lib

  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(input_graph, "rb") as f:
    data = f.read()
    if frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_names.split(","),
      output_names.split(","), dtypes.string.as_datatype_enum)

  if frozen_graph:
    f = gfile.FastGFile(output_graph, "w")
    f.write(output_graph_def.SerializeToString())
  else:
    graph_io.write_graph(output_graph_def,
                         os.path.dirname(output_graph),
                         os.path.basename(output_graph))


def pb_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.

  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  import_pb_to_tensorboard.import_to_tensorboard(
      model_dir=model_dir,
      log_dir=log_dir)


def ckpt_change_prefix(infile, outfile, old_prefix, new_prefix):
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


def ckpt_compute_parameter(filepath):
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
