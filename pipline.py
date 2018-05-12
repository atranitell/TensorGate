# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
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
# ==============================================================================
"""pipline"""
import os
import argparse
import json


def run(filepath, gpu):
  """ data format should be like this:
  [ { "task": "kinface.vae",
      "extra": {
        "name": "kinface2.vae",
        "target": "kinvae.encoder",
        "train": {"entry_path": "train_1.txt"},
        "test": {"entry_path": "test_1.txt"},
        "val": {"entry_path": "train_1.txt"}}},
    { "task": "kinface.vae",
      "extra": {
        "name": "kinface2.vae",
        "target": "kinvae.encoder",
        "train": {"entry_path": "train_2.txt"},
        "test": {"entry_path": "test_2.txt"},
        "val": {"entry_path": "train_2.txt"}}}]
  """
  with open(filepath) as fp:
    content = json.load(fp)

  random_file = '_df8192jkfjsDF.json'
  for item in content:
    with open(random_file, 'w') as fw:
      json.dump(item['extra'], fw)
    cmd = 'python main.py %s -dataset=%s -config=%s' % (
        gpu, item['task'], random_file)
    os.system(cmd)
    os.remove(random_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-file', type=str, dest='file')
  parser.add_argument('-gpu', type=str, dest='gpu', default='0')
  args, _ = parser.parse_known_args()
  run(args.file, args.gpu)
