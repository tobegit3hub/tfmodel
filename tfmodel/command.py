#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Copyright 2017 The Authors. All Rights Reserved.
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

import argparse
import logging
import pkg_resources
import sys
import coloredlogs

from tfmodel.savedmodel_analyst import SavedmodelAnalyst

coloredlogs.install()
logging.basicConfig(level=logging.DEBUG)


def validate_model(args):
  logging.info("Try to validate the model: {}".format(args.model))

  savedmodel = SavedmodelAnalyst(args.model)

  is_validated = savedmodel.validate()
  if is_validated:
    logging.info("Yes, it is the validated model")
  else:
    logging.error("False, it is not validated model")


def inspect_model(args):
  logging.info("Try to inspect the model: {}".format(args.model))

  savedmodel = SavedmodelAnalyst(args.model)

  savedmodel.print_signature()


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "-v",
      "--version",
      action="version",
      version=pkg_resources.require("tensorflow-model")[0].version,
      help="Display sdk version")

  main_subparser = parser.add_subparsers(dest="command_group", help="Commands")

  # subcommand: validate
  validate_parser = main_subparser.add_parser("validate")
  validate_parser.add_argument("model", help="Path of the model")
  validate_parser.set_defaults(func=validate_model)

  # subcommand: inspect
  inspect_parser = main_subparser.add_parser("inspect")
  inspect_parser.add_argument("model", help="Path of the model")
  inspect_parser.set_defaults(func=inspect_model)
  """
  # subcommand: study describe
  study_describe_parser = study_subparser.add_parser(
      "describe", help="Describe studiy")
  study_describe_parser.add_argument(
      "-s",
      "--study_name",
      dest="study_name",
      help="The id of the resource",
      required=True)
  study_describe_parser.set_defaults(func=describe_studie)
  """

  # Display help information by default
  if len(sys.argv) == 1:
    args = parser.parse_args(["-h"])
  else:
    args = parser.parse_args(sys.argv[1:])
  args.func(args)


if __name__ == "__main__":
  main()
