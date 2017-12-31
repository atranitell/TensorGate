""" a pipline for processing task
"""
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
    cmd = 'python gate.py %s -name=%s -extra=%s' % (
        gpu, item['task'], random_file)
    os.system(cmd)
    os.remove(random_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-file', type=str, dest='file')
  parser.add_argument('-gpu', type=str, dest='gpu')
  args, _ = parser.parse_known_args()
  run(args.file, args.gpu)
