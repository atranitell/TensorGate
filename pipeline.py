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
import uuid


def run(argments):
  """ data format should be like this:
  [ 
    { 
      "dataset": "kinface.vae",
      "run": true,
      "config": {
        "name": "kinface2.vae",
        "target": "kinvae.encoder",
        "train": {"entry_path": "train_1.txt"},
        "test": {"entry_path": "test_1.txt"},
        "val": {"entry_path": "train_1.txt"}
      }
    },
    { 
      "dataset": "kinface.vae",
      "run": false,
      "config": {
        "name": "kinface2.vae",
        "target": "kinvae.encoder",
        "train": {"entry_path": "train_2.txt"},
        "test": {"entry_path": "test_2.txt"},
        "val": {"entry_path": "train_2.txt"}
      }
    }
  ]
  """
  # mkdir to store temp file
  if not os.path.exists('_tmp'):
    os.mkdir('_tmp')
  # load config.pipline for processing
  with open(argments.file) as fp:
    content = json.load(fp)
  # there, we split config for a several subtask for the aim of releasing
  #   tensorflow source fully.
  for task in content:
    # if executable
    if 'run' in task and task['run'] == False:
      continue
    # write a temp file
    random_file = '_tmp/' + uuid.uuid4().hex + '.json'
    with open(random_file, 'w') as fw:
      json.dump(task['config'], fw)
    # exec
    main = 'main.py' if os.path.exists('main.py') else 'main.pyc'
    cmd_str = 'python %s -gpu=%s -dataset=%s -config=%s'
    cmd = cmd_str % (main, argments.gpu, task['dataset'], random_file)
    print(cmd)
    os.system(cmd)
    os.remove(random_file)
    print('\n\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-file', type=str, dest='file')
  parser.add_argument('-gpu', type=str, dest='gpu', default='0')
  args, _ = parser.parse_known_args()
  run(args)
