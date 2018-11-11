import os
import logging
import time
import numpy as np
import tensorflow as tf
from prettytable import PrettyTable

from utils import ModelUtil


class SavedmodelAnalyst(object):
  """
  The helper class to access TensorFlow Savedmodel.
  """

  def __init__(self, savedmodel_path):
    """
    Get the base model path and get the model versions.
    """

    self.savedmodel_path = savedmodel_path

    is_model_directory_exist = os.path.isdir(self.savedmodel_path)

    # Check if the directory exists or not
    if is_model_directory_exist:
      # Get model version, example: ["1", "2"]
      self.model_version_list = os.listdir(self.savedmodel_path)
      logging.info("Get model versions: {}".format(self.model_version_list))

      # Use the first model version, example: "./model/1"
      self.model_version_path = os.path.join(self.savedmodel_path,
                                             self.model_version_list[0])

      self.model_file_exist = True

    else:
      # Set false if model does not exist
      self.model_file_exist = False
      logging.error("The model path does not exist: {}".format(
          self.savedmodel_path))

  def validate(self):
    """
    Validate the model.
    """

    if self.model_file_exist == False:
      return False

    try:
      session = tf.Session(graph=tf.Graph())
      tf.saved_model.loader.load(session,
                                 [tf.saved_model.tag_constants.SERVING],
                                 self.model_version_path)
      return True
    except Exception as e:
      logging.error("Fail to validate and get error: {}".format(e))
      return False

  def inspect_model(self):
    """
    Inspect the model to print the model signature.    
    """

    if self.validate() == False:
      logging.error("Fail to load the model")
      return

    try:
      session = tf.Session(graph=tf.Graph())
      meta_graph = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING],
          self.model_version_path)
      logging.info("Succeed to load model in: {}".format(
          self.model_version_path))

    except Exception as e:
      logging.info("Fail to inspect model and error: {}".format(e))

    # Print the model signature
    signature_name = list(meta_graph.signature_def.items())[0][0]
    model_graph_signature = list(meta_graph.signature_def.items())[0][1]

    logging.info("Print the signature of the model")
    logging.info("Model signature name: {}, method: {}".format(
        signature_name, model_graph_signature.method_name))

    # Print the table of input tensors
    table = PrettyTable()
    table.field_names = ["InputName", "OpName", "DType", "Shape"]

    for item in model_graph_signature.inputs.items():
      """ Example:
      ( 'features', 
        name: "Placeholder:0"
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 9
          }
        }
      )
      """
      name = item[0]
      op_name = item[1].name
      """ TODO: refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
      DT_FLOAT = 1;
      DT_DOUBLE = 2;
      DT_INT32 = 3;
      DT_UINT8 = 4;
      DT_INT16 = 5;
      DT_INT8 = 6;
      DT_STRING = 7;
      DT_COMPLEX64 = 8;
      DT_INT64 = 9;
      DT_BOOL = 10;
      """
      dtype_int = item[1].dtype
      if dtype_int == 1:
        dtype = "DT_FLOAT"
      elif dtype_int == 2:
        dtype = "DT_DOUBLE"
      elif dtype_int == 3:
        dtype = "DT_INT32"
      elif dtype_int == 4:
        dtype = "DT_UINT8"
      elif dtype_int == 5:
        dtype = "DT_INT16"
      elif dtype_int == 6:
        dtype = "DT_INT8"
      elif dtype_int == 7:
        dtype = "DT_STRING"
      elif dtype_int == 8:
        dtype = "DT_COMPLEX64"
      elif dtype_int == 9:
        dtype = "DT_INT64"
      elif dtype_int == 10:
        dtype = "DT_BOOL"
      else:
        dtype = dtype_int
      # Example: [-1, 9]
      shape = [dim.size for dim in item[1].tensor_shape.dim]
      #logging.info("Name: {}, Op: {}, Type: {}, Shape: {}".format(name, op_name, dtype, shape))

      table.add_row([name, op_name, dtype, shape])

    print(table)

    # Print the table of output tensors
    table = PrettyTable()
    table.field_names = ["OutputName", "OpName", "DType", "Shape"]

    for item in model_graph_signature.outputs.items():
      """ Example:
      ( 'softmax', 
        name: "Softmax_2:0"
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 2
          }
        }
      )
      """
      name = item[0]
      op_name = item[1].name
      dtype_int = item[1].dtype
      if dtype_int == 1:
        dtype = "DT_FLOAT"
      elif dtype_int == 2:
        dtype = "DT_DOUBLE"
      elif dtype_int == 3:
        dtype = "DT_INT32"
      elif dtype_int == 4:
        dtype = "DT_UINT8"
      elif dtype_int == 5:
        dtype = "DT_INT16"
      elif dtype_int == 6:
        dtype = "DT_INT8"
      elif dtype_int == 7:
        dtype = "DT_STRING"
      elif dtype_int == 8:
        dtype = "DT_COMPLEX64"
      elif dtype_int == 9:
        dtype = "DT_INT64"
      elif dtype_int == 10:
        dtype = "DT_BOOL"
      else:
        dtype = dtype_int
      # Example: [-1, 9]
      shape = [dim.size for dim in item[1].tensor_shape.dim]

      table.add_row([name, op_name, dtype, shape])

    print(table)

  def benchmark_model_with_mock_data(self):
    """
    Generate mock data to benchmark the model and print performance.
    """

    if self.validate() == False:
      logging.error("Fail to load the model")
      return

    try:
      session = tf.Session(graph=tf.Graph())
      meta_graph = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING],
          self.model_version_path)
      logging.info("Succeed to load model in: {}".format(
          self.model_version_path))

    except Exception as e:
      logging.info("Fail to benchmark model and error: {}".format(e))

    # Get the model signature
    model_graph_signature = list(meta_graph.signature_def.items())[0][1]

    # Generate output op names for infernece
    output_op_names = []
    for item in model_graph_signature.outputs.items():
      output_op_name = item[1].name
      output_op_names.append(output_op_name)

    input_items = model_graph_signature.inputs.items()

    batch_size_list = [1, 10, 1000, 10000, 100000]

    for batch_size in batch_size_list:
      feed_dict_map = ModelUtil.construct_feed_dict_with_batch(
          input_items, batch_size)

      #logging.info("Generate input feed_dict: {}".format(feed_dict_map))

      start_time = time.time()
      # Example: [array([[1., 1.], [1., 1.]], dtype=float32), array([[1], [1]], dtype=int32), array([1, 1])]
      result_ndarrays = session.run(output_op_names, feed_dict=feed_dict_map)

      inference_time = time.time() - start_time
      qps = 1.0 / inference_time

      #logging.info("Inference resuolt: {}".format(result_ndarrays))
      logging.info("Inference batch size: {}, time: {}s, qps: {}".format(
          batch_size, inference_time, qps))

  def export_tensorboard_files(self, tensorboard_path):
    """
    Read the model and export the TensorBoard files.
    """

    if self.validate() == False:
      logging.error("Fail to load the model")
      return

    try:
      graph = graph = tf.Graph()
      session = tf.Session()

      #merged = tf.summary.merge_all()

      meta_graph = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING],
          self.model_version_path)
      logging.info("Succeed to load model in: {}".format(
          self.model_version_path))

      tensorboard_writer = tf.summary.FileWriter(tensorboard_path,
                                                 session.graph)
      #tensorboard_writer.add_summary(summary)
      logging.info("Write tensorboard")

    except Exception as e:
      logging.info("Fail to benchmark model and error: {}".format(e))

    # Get the model signature
    model_graph_signature = list(meta_graph.signature_def.items())[0][1]

    # Generate output op names for infernece
    output_op_names = []
    for item in model_graph_signature.outputs.items():
      output_op_name = item[1].name
      output_op_names.append(output_op_name)

    input_items = model_graph_signature.inputs.items()

    #tf.summary.scalar('test', tf.constant(1))
    #merged = tf.summary.merge_all()
    """
    feed_dict_map = ModelUtil.construct_feed_dict_with_batch(
            input_items, 1)

    #logging.info("Generate input feed_dict: {}".format(feed_dict_map))
    # Example: [array([[1., 1.], [1., 1.]], dtype=float32), array([[1], [1]], dtype=int32), array([1, 1])]
    #result_ndarrays = session.run(output_op_names, feed_dict=feed_dict_map)
    """

    #summary, _= session.run([merged, output_op_names], feed_dict=feed_dict_map)
    """
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    session.run(tf.global_variables_initializer())


    summary, _= session.run([merged, output_op_names],
                            feed_dict=feed_dict_map,
                            options=run_options,
                            run_metadata=run_metadata)


    tensorboard_writer.add_run_metadata(run_metadata, 'step%d' % 1)
    tensorboard_writer.add_summary(summary, 1)
    """

    logging.info(
        "Success to export the tensorboard files: {}".format(tensorboard_path))
