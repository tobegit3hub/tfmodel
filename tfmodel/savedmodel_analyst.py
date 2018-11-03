
import logging
import tensorflow as tf


class SavedmodelAnalyst(object):
  def __init__(self, savedmodel_path):
    self.savedmodel_path = savedmodel_path

  def print_signature(self):
    logging.info("Print the signature of model")
    #import ipdb;ipdb.set_trace()
