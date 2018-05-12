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
"""Offer a set of analysis tools

The module is independent of TensorGate, its workspace is determined by input
file path. 

It accept such inputs:
- the config file
- the folder with several config files.

The config file type:
- template (generate config file)
- config file (stands for a task)

Current functions:
- draw curve with multi-line
- draw roc curve
- compute statistic value

"""

import argparse
import utils
import chart_line
import kv_stat


def interface(args):
  """ a command manager tool """
  config = utils.load_json(args.file)
  if config['task'] == 'basic_line_chart':
    chart_line.draw_basic_line_chart(config)
  elif config['task'] == 'basic_line_chart_template':
    chart_line.draw_basic_line_chart_template(config)
  elif config['task'] == 'compute_kv':
    kv_stat.compute_kv(config)
  elif config['task'] == 'compute_kv_template':
    kv_stat.compute_kv_template(config)
  elif config['task'] == 'compute_kv_template_folder':
    kv_stat.compute_tv_template_folder(config)
    # elif config"['task']" == 'basic_line_chart_template':
    #   _config.gen_line_config(config, root)
    # elif config"['task']" == 'roc':
    #   _roc.draw_roc(config)
    # elif config"['task']" == 'trend':
    #   _trend.draw_trend(config)
    # elif config"['task']" == 'stat':
    #   _statistic.print_info(config, root)
  else:
    raise ValueError('Unknown input task [%s].', config['task'])


if __name__ == "__main__":
  PARSER = argparse.ArgumentParser()
  PARSER.add_argument('-file', type=str, default=None, dest='file')
  PARSER.add_argument('-fold', type=str, default=None, dest='fold')
  ARGS, _ = PARSER.parse_known_args()
  interface(ARGS)
