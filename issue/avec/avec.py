""" Task Set for AVEC2014
"""
from issue.avec.avec_audio_cnn import AVEC_AUDIO_CNN
from issue.avec.avec_image_cnn import AVEC_IMAGE_CNN

def select(config):
  """ select different subtask
  """
  if config.target == 'avec.audio.cnn':
    return AVEC_AUDIO_CNN(config)
  elif config.target == 'avec.image.cnn':
    return AVEC_IMAGE_CNN(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
