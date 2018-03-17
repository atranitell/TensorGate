from issue.lab.guided_learning import GUIDED_LEARNING

def select(config):
  """ select different subtask
  """
  if config.target == 'lab.guided-learning':
    return GUIDED_LEARNING(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
