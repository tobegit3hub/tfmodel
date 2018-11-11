#!/bin/bash

set -e
set -x

tfmodel validate ./model
tfmodel inspect ./model
tfmodel benchmark ./model
tfmodel tensorboard ./model
