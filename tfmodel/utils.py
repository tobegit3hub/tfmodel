import numpy as np
import tensorflow as tf


class ModelUtil(object):
  """
  The utils class for TensorFlow models with static methods.
  """

  @staticmethod
  def construct_feed_dict_with_batch(input_items, batch_size=1):
    #batch_size = 100000

    # Generate feed dict data for inference, example: {u'Softmax_2:0': [[1.0, 1.0], [1.0, 1.0]], u'Identity:0': [[1], [1]], u'ArgMax_2:0': [1, 1]}
    feed_dict_map = {}

    for item in input_items:
      # Example: "Placeholder_0"
      input_op_name = item[1].name

      shape_list = [dim.size for dim in item[1].tensor_shape.dim]
      shape_number = len(shape_list)

      dtype = item[1].dtype

      # Use to generated the nested array
      internal_array = None

      # Travel all the dims in reverse order
      for i in range(shape_number):
        dim = shape_list[shape_number - 1 - i]

        if dim == -1:
          dim = batch_size

        if internal_array == None:
          # Fill with default values by the types, refer to https://www.tensorflow.org/api_docs/python/tf/DType
          if dtype == int(tf.int8) or dtype == int(tf.uint8) or dtype == int(
              tf.int16) or dtype == int(tf.uint16) or dtype == int(
                  tf.int32) or dtype == int(tf.uint32) or dtype == int(
                      tf.int64) or dtype == int(tf.uint64):
            value = 1
          elif dtype == int(tf.bool):
            value = True
          elif dtype == int(tf.string):
            value = "A"
          else:
            value = 1.0

          # Example: [1.0, 1.0]
          internal_array = [value for i in range(dim)]
          #internal_array = np.asarray(internal_array)

        else:
          # Example: [[1.0, 1.0]]
          internal_array = [internal_array for i in range(dim)]

      # TODO: Generate the numpy ndarray before using Python list
      internal_array = np.asarray(internal_array)
      feed_dict_map[input_op_name] = internal_array

    return feed_dict_map
