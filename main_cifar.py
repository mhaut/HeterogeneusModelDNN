# coding=utf-8
# Copyright 2019 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST using Mesh TensorFlow and TF Estimator.

This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import mesh_tensorflow as mtf
from cifar_dataset import cifar_dset# as dataset  # local file import
import tensorflow.compat.v1 as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.get_logger().setLevel('WARNING')


tf.flags.DEFINE_string("data_dir", "/tmp/cifar_data",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", "/tmp/cifar_model", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 200,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
tf.flags.DEFINE_integer("train_epochs", 30, "Total number of training epochs.")
tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", "b1:3;b2:1;b3:1;b4:1;b5:1;b6:1;b7:1;b8:1", "mesh shape")
tf.flags.DEFINE_string("layout", "filters1:b1;filters2:b2;filters3:b3;filters4:b4;filters5:b5;filters6:b6;filters7:b7;filters8:b8",
                       "layout rules")
tf.app.flags.DEFINE_string("list_speed", "34,33,33","list in string separated by comma")

FLAGS = tf.flags.FLAGS


def cifar_model(image, labels, mesh):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  row_blocks_dim = mtf.Dimension("row_blocks", 1)
  col_blocks_dim = mtf.Dimension("col_blocks", 1)
  rows_dim = mtf.Dimension("rows_size", 32)
  cols_dim = mtf.Dimension("cols_size", 32)
  init = 60

  classes_dim = mtf.Dimension("classes", 10)
  one_channel_dim = mtf.Dimension("one_channel", 3)

  x = mtf.import_tf_tensor(
      mesh, tf.reshape(image, [FLAGS.batch_size, 1, 32, 1, 32, 3]),
      mtf.Shape(
          [batch_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, one_channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, row_blocks_dim, col_blocks_dim,
      rows_dim, cols_dim, one_channel_dim])

  # add some convolutional layers to demonstrate that convolution works.
  filters1_dim = mtf.Dimension("filters1", init)
  filters2_dim = mtf.Dimension("filters2", init)
  f1 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters1_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv0"))
  #print("conv:, ", f1.shape)

  f2 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f1, filters2_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv1"))

  x = mtf.layers.max_pool2d(f2, ksize=(2,2), name="maxpool0")

  #print(x.shape)

  filters3_dim = mtf.Dimension("filters3", init*2)
  filters4_dim = mtf.Dimension("filters4", init*2)

  f3 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters3_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv2"))
  f4 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f3, filters4_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv3"))

  x = mtf.layers.max_pool2d(f4, ksize=(2,2), name="maxpool1")

  #print(x.shape)
  filters5_dim = mtf.Dimension("filters5", init*4)
  filters6_dim = mtf.Dimension("filters6", init*4)

  f5 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters5_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv4"))
  f6 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f5, filters6_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv5"))

  x = mtf.layers.max_pool2d(f6, ksize=(2,2), name="maxpool2")
  #print(x.shape)

  filters7_dim = mtf.Dimension("filters7", init*8)
  filters8_dim = mtf.Dimension("filters8", init*8)

  f7 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters7_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv6"))
  f8 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f7, filters8_dim, filter_size=[3, 3], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv7"))

  x = mtf.layers.max_pool2d(f8, ksize=(2,2), name="maxpool3")
#  x = mtf.reduce_mean(f8, reduced_dim=filters8_dim)
  # add some fully-connected dense layers.
  #hidden_dim1 = mtf.Dimension("hidden1", init*8)
  hidden_dim1 = mtf.Dimension("hidden1", 256)
  hidden_dim2 = mtf.Dimension("hidden2", init*8)

  h1 = mtf.layers.dense(
      x, hidden_dim1,
      reduced_dims=x.shape.dims[-5:],
      activation=mtf.relu, name="hidden1")
  #h2 = mtf.layers.dense(
      #h1, hidden_dim2,
      #activation=mtf.relu, name="hidden2")
  logits = mtf.layers.dense(h1, classes_dim, name="logits")
  if labels is None:
    loss = None
  else:
    labels = mtf.import_tf_tensor(
        mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, mtf.one_hot(labels, classes_dim), classes_dim)
    loss = mtf.reduce_mean(loss)

  all_filters = [[init, init, init*2, init*2, init*4, init*4, init*8, init*8]]
  return logits, loss, all_filters


def heterogeneousPartition(blocks):

  #part_file = "path/part.dist"
  #speeds, dev = readfile(part_file)
  #speeds = [0.45, 0.45, 0.10]
  speeds = [float(item) / 100.0 for item in FLAGS.list_speed.split(',')]
  num_input=0
  block_elements = 3
  all_filters = []
  w_block=0
  for num_fil in blocks:
    num_input=0
    for j in num_fil:
      filters = []
      if w_block%2==0:
        [filters.append(int(j*speeds[i])) for i in range(0,len(speeds))]
      else:
        [filters.append(j) for i in range(0,len(speeds))]

      filters[np.argmin(filters)]=1 if(filters[np.argmin(filters)] < 1) else filters[np.argmin(filters)]
      if w_block%2==0:
        while sum(filters) != num_fil[num_input]:
          #print(sum(filters), num_fil[num_input])
          filters[np.argmax(filters)]+=1
      #print(filters)
      all_filters.append(filters)
      num_input+=1
      w_block+=1
  all_filters_blocks = []
  for i in range(0,len(all_filters)-1, 8):
    all_filters_blocks.append([all_filters[i], all_filters[i+1], all_filters[i+2], all_filters[i+3], all_filters[i+4], all_filters[i+5], all_filters[i+6], all_filters[i+7]])

  #print(all_filters_blocks)
  return speeds, all_filters_blocks

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  tf.logging.info("features = %s labels = %s mode = %s params=%s" %
                  (features, labels, mode, params))
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  logits, loss, all_filters = cifar_model(features, labels, mesh)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  mesh_size = mesh_shape.size
  speeds, conv_shape = heterogeneousPartition(all_filters)
  #print(conv_shape)
  mesh_devices = ["gpu:0", "gpu:1", "cpu:0"]
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices, conv_shape, all_filters, speeds)

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  restore_hook = mtf.MtfRestoreHook(lowering)

  tf_logits = lowering.export_to_tf_tensor(logits)
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf.summary.scalar("loss", tf_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)
    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=2,
        defer_build=False, save_relative_paths=True)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    saver_listener = mtf.MtfCheckpointSaverListener(lowering)
    saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.model_dir,
        save_steps=1000,
        saver=saver,
        listeners=[saver_listener])

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(tf_logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(tf_loss, "cross_entropy")
    tf.identity(accuracy[1], name="train_accuracy")

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar("train_accuracy", accuracy[1])

    # restore_hook must come before saver_hook
    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
        training_chief_hooks=[restore_hook, saver_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(tf_logits, axis=1),
        "probabilities": tf.nn.softmax(tf_logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[restore_hook],
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf_loss,
        evaluation_hooks=[restore_hook],
        eval_metric_ops={
            "accuracy":
            tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
        })


def run_cifar():
  """Run MNIST training and eval loop."""
  cifar_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir)
  dataset = cifar_dset()

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(FLAGS.data_dir)
    ds_batched = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds_batched.repeat(FLAGS.epochs_between_evals)
    return ds

  def eval_input_fn():
    return dataset.test(FLAGS.data_dir).batch(
        FLAGS.batch_size).make_one_shot_iterator().get_next()

  # Train and evaluate model.
  import time
  time_tot_start = 0
  time_epoch_start = 0
  time_tot_start = time.time()
  f = open("./Het_CNN.txt", "a+")
  f.write("#Filters\t#Epochs\t#Time\t#Accuracy\t#Loss\t#Shape\n")
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  mesh_size = mesh_shape.size
  conv_shape = []

  for ep in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    time_epoch_start = time.time()
    cifar_classifier.train(input_fn=train_input_fn, hooks=None)
    time_epoch_end = time.time()-time_epoch_start
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print("\nEvaluation results:\n\t%s\n" % eval_results)
    print(ep, "----------->", time_epoch_end)
    f.write("%d\t%0.6f\t%0.6f\t%0.6f\t%s\n" % (ep, time_epoch_end, eval_results['accuracy'], eval_results['loss'], conv_shape))

  time_tot_end = time.time()-time_tot_start
  print("Total Time ", FLAGS.train_epochs, " Epochs", time_tot_end)

  f.close()

def main(_):
  run_cifar()


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
