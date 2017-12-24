"""
"""
import os
import argparse

tasklist = {
    '1': ['kinvae1.pair', 'kinvae1.pair2', 'kinvae1.pair3', 'kinvae1.pair4', 'kinvae1.pair5'],
    '2': ['kinvae2.pair', 'kinvae2.pair2', 'kinvae2.pair3', 'kinvae2.pair4', 'kinvae2.pair5']
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-name', type=str, dest='name')
  parser.add_argument('-gpu', type=str, dest='gpu')
  args, _ = parser.parse_known_args()

  for name in tasklist[args.name]:
    os.system('python gate.py %s -name=%s' % (args.gpu, name))
