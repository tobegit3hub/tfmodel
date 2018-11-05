
import os
import logging
import tensorflow as tf


class SavedmodelAnalyst(object):
  def __init__(self, savedmodel_path):
    self.savedmodel_path = savedmodel_path

  def print_signature(self):
    logging.info("Print the signature of model")
    #import ipdb;ipdb.set_trace()

    # TODO: Support hdfs path with tensorflow gfile api
    model_version_list = os.listdir(self.savedmodel_path)
    logging.info("Model versions: {}".format(model_version_list))


    # TODO: Use all the model versions
    # Example: "./model/1"
    model_version_path = os.path.join(self.savedmodel_path, model_version_list[0])


    try:
      session = tf.Session(graph=tf.Graph())
      meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_version_path)
      logging.info("Succeed to load model in: {}".format(model_version_path))

      import ipdb;ipdb.set_trace()
      signature_name = list(meta_graph.signature_def.items())[0][0]
      model_graph_signature = list(meta_graph.signature_def.items())[0][1]


    except IOError as ioe:
      logging.info("Fail to load model and catch exception: {}".format(ioe))