# Copyright 2017.
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

# python ./setup.py sdist --format=gztar
# twine upload dist/tensorflow-model-x.x.x.tar.gz


try:
  from setuptools import setup
  setup()
except ImportError:
  from distutils.core import setup

# TODO: Remove the code of try-except
from setuptools import setup, find_packages

setup(
    name="tensorflow-model",
    version="0.0.1",
    author="tobe",
    author_email="tobeg3oogle@gmail.com",
    url="https://github.com/tobegit3hub/tfmodel",
    install_requires=["tensorflow"],
    description="Command-line tool to inspect TensorFlow models",
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "tfmodel=tfmodel.command:main",
        ],
    })
