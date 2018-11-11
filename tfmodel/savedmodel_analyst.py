import os
import logging
import tensorflow as tf


class SavedmodelAnalyst(object):
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

  def print_signature(self):
    logging.info("Print the signature of model")
    #import ipdb;ipdb.set_trace()

    if self.model_file_exist == False or self.validate() == False:
      logging.error("Fail to load savedmodel")

    # TODO: Support hdfs path with tensorflow gfile api
    model_version_list = os.listdir(self.savedmodel_path)
    logging.info("Model versions: {}".format(model_version_list))

    # TODO: Use all the model versions
    # Example: "./model/1"
    model_version_path = os.path.join(self.savedmodel_path,
                                      model_version_list[0])

    try:
      session = tf.Session(graph=tf.Graph())
      meta_graph = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], model_version_path)
      logging.info("Succeed to load model in: {}".format(model_version_path))

      #import ipdb;ipdb.set_trace()
      signature_name = list(meta_graph.signature_def.items())[0][0]
      model_graph_signature = list(meta_graph.signature_def.items())[0][1]

    except IOError as ioe:
      logging.info("Fail to load model and catch exception: {}".format(ioe))
