#!/bin/bash

set -e
set -x

tfmodel validate ./examples/model
tfmodel inspect ./examples/model
tfmodel benchmark ./examples/model
tfmodel tensorboard ./examples/model
