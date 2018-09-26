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
"""FOR AVEC2014 SERIES"""

from samples.avec2014.avec2014_audio_cnn import AVEC2014_AUDIO_CNN
from samples.avec2014.avec2014_img_cnn import AVEC2014_IMG_CNN
from samples.avec2014.avec2014_img_cnn_dldl import AVEC2014_IMG_CNN_DLDL
from samples.avec2014.avec2014_68_img_cnn import AVEC2014_68_IMG_CNN
from samples.avec2014.avec2014_img_4view import AVEC2014_IMG_4VIEW
from samples.avec2014.avec2014_img_bicnn import AVEC2014_IMG_BICNN
from samples.avec2014.avec2014_audio_fcn import AVEC2014_AUDIO_FCN
from samples.avec2014.avec2014_audio_vae import AVEC2014_AUDIO_VAE
from samples.avec2014.avec2014_audio_global import AVEC2014_AUDIO_GLOBAL

def select(config):
  """ select different subtask
  """
  if config.target == 'avec2014.img.cnn':
    return AVEC2014_IMG_CNN(config)
  elif config.target == 'avec2014.img.cnn.dldl':
    return AVEC2014_IMG_CNN_DLDL(config)
  elif config.target == 'avec2014.68.img.cnn':
    return AVEC2014_68_IMG_CNN(config)
  elif config.target == 'avec2014.img.4view':
    return AVEC2014_IMG_4VIEW(config)
  elif config.target.startswith('avec2014.img.bicnn'):
    return AVEC2014_IMG_BICNN(config)
  elif config.target == 'avec2014.audio.cnn':
    return AVEC2014_AUDIO_CNN(config)
  elif config.target == 'avec2014.audio.fcn':
    return AVEC2014_AUDIO_FCN(config)
  elif config.target == 'avec2014.audio.vae':
    return AVEC2014_AUDIO_VAE(config)
  elif config.target == 'avec2014.audio.global':
    return AVEC2014_AUDIO_GLOBAL(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
