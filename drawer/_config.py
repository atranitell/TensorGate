""" A tool helps to genertate config file from a folder with logs
"""

import os
import utils


def gen_line_config(config, root):
  # relative to .json file
  config['info']['dir'] = os.path.join(root, config['info']['dir'])

  # copy to a new memory, avoid lost info
  _i = config['info'].copy()
  _f = config['figure'].copy()
  _d = config['data'].copy()

  # fill the figure information
  if _f['title'] is None:
    config['figure']['title'] = os.path.basename(_i['dir'])
  if _f['save_fig'] is None:
    config['figure']['save_fig'] = os.path.join(
        _i['dir'], os.path.basename(_i['dir']) + '.png')

  # fill in the data
  if len(_d) < 1:
    raise ValueError('A template data is essential!')

  config['data'] = []
  for log_name in os.listdir(_i['dir']):
    if log_name.split('.')[-1] != 'log':
      continue
    config['data'].append(_d[0].copy())
    config['data'][-1]['legend'] = log_name
    config['data'][-1]['path'] = os.path.join(_i['dir'], log_name)

  # output to jsonfile
  output = os.path.join(_i['dir'], os.path.basename(_i['dir']) + '.json')

  # change type
  config['figure']['type'] = 'basic_line_chart'
  config.pop('info')

  utils.save_json(config, output)
  print('Config file has been saved in %s' % output)

  if _i['display']:
    os.system('python main.py -file=%s' % output)
