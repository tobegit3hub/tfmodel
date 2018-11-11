import os
import logging
import tensorflow as tf

from prettytable import PrettyTable


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

    logging.info("Try to inspect the model")

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



  def inference_model_with_mock_data(self):
    logging.info("Try to inference the model with mock data")

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