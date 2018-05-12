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

import os
import matplotlib.pyplot as plt
import utils
import data_parser


def draw_basic_line_chart(config):
  """Parse log data and calling draw"""
  draw_data_pairs = []
  for _cfg in config['data']:
    data = data_parser.log_kv(_cfg['path'], _cfg['phase'], _cfg['keys'])

    # downsampling all points including iter
    if _cfg['invl'] > 1:
      utils.process_keys(utils.downsampling, data, _cfg['invl'])

    # smooth all data[key] except for iter
    if _cfg['smooth'] > 1:
      data = utils.process_keys(
          utils.smooth, data, _cfg['smooth'], ignore_keys=['iter'])

    # add each key into draw data
    pairs = []
    for key in _cfg['keys']:
      pairs.append((data['iter'], data[key]))
    draw_data_pairs.append(pairs)

  # start to draw
  draw(config, draw_data_pairs)


def draw(config, groups: list):
  """Draw a line chart.
    figure attribution:
      'title', 'xlabel', 'ylabel', 'xmax', 'xmin', 'ymax', 'ymin'
      'save_fig'
    line attribution:
      'color', 'alpha', 'legend'

  Args:
    config: config['figure']<dict>, config['data']<list<dict>>
    collections: [pairs1, pairs2, ...]
    paris: a list including serveral tuples:
        [(x<list>, y<list>), (x1<list>, y1<list>), ...]
        x: a int list. e.g., data['iter']
        y: a float list. e.g. data[key]
  """
  plt.close('all')
  _fig = config['figure']
  _lines = config['data']

  def fill(key, default, fig=_fig):
    """Fill key if existing else default"""
    return fig[key] if key in fig else default

  for i, paris in enumerate(groups):
    for j, pair in enumerate(paris):
      color = fill('color', None, _lines[i])
      style = fill('style', '-', _lines[i])
      label = fill('legend', None, _lines[i]) + ' ' + _lines[i]['keys'][j]
      alpha = fill('alpha', 0.8, _lines[i])
      plt.plot(pair[0], pair[1], style, color=color, label=label, alpha=alpha)

  plt.grid()

  # label
  plt.title(fill('title', 'Line chart'))
  plt.xlabel(fill('xlabel', 'iter'))
  plt.ylabel(fill('ylabel', 'value'))

  # lim
  plt.xlim(xmin=fill('xmin', 0))
  plt.xlim(xmax=fill('xmax', None))
  plt.ylim(ymin=fill('ymin', None))
  plt.ylim(ymax=fill('ymax', None))

  # show legend
  plt.legend()

  # save
  if 'save_fig' in _fig:
    plt.savefig(_fig['save_fig'])

  # plt show
  plt.show()


def draw_basic_line_chart_template(config):
  """Draw from a line template"""
  # copy to a new memory, avoid lost info
  _i = config['info'].copy()
  _f = config['figure'].copy()
  _d = config['data'].copy()

  # fill the figure information
  if _f['title'] is None:
    config['figure']['title'] = os.path.basename(_i['dir'])
  if _f['save_fig'] is None:
    save_fig = os.path.basename(_i['dir']) + '.png'
    config['figure']['save_fig'] = os.path.join(_i['dir'], save_fig)

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
  output_filename = os.path.basename(_i['dir'])+'.'+_i['task'] + '.json'
  output = os.path.join(_i['dir'], output_filename)

  # change type
  config['task'] = _i['task']
  config.pop('info')

  utils.save_json(config, output)
  print('Config file has been saved in %s' % output)

  if _i['run']:
    draw_basic_line_chart(config)
