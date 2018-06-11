"""TEST BEST MODEL"""

import os
import re

_TEST_DIR = '../_outputs'

# folder_path
# best iter, mae, rmse
res = []

for fold in os.listdir(_TEST_DIR):
  fold_path = os.path.join(_TEST_DIR, fold)

  for fname in sorted(os.listdir(fold_path)):
    if fname.find('.log') > 0:
      fpath = os.path.join(fold_path, fname)
      break

  if fpath.find('.log') < 0:
    continue

  with open(fpath, 'r') as fp:
    _min_iter = '0'
    _min_rmse = 100.0
    _min_mae = 100.0
    for line in fp:
      if line.find('[VAL]') > 0 and line.find('video_rmse') > 0:
        _iter = (re.findall('Iter:(.*?),', line)[0])
        _mae = float(re.findall('video_mae:(.*?),', line)[0])
        _rmse = float(re.findall('video_rmse:(.*)', line)[0])
        if _rmse < _min_rmse:
          _min_mae = _mae
          _min_rmse = _rmse
          _min_iter = _iter

    print(fold_path)
    print('  Iter: %s, mae: %.4f, rmse: %.4f\n' %
          (_min_iter, _min_mae, _min_rmse))

    res.append((fold_path, _min_iter, _min_rmse, _min_mae))

for model in res:
  ckpt_path = os.path.join(model[0], 'checkpoint')
  content = ''
  with open(ckpt_path, 'r') as fp:
    for line in fp:
      if line.find('model_checkpoint_path:') >= 0:
        line = line.replace(re.findall('.ckpt-(.*?)\"', line)[0], model[1])
        print('Load:', line)
      content += line
  with open(ckpt_path, 'w') as fw:
    fw.write(content)

  main = 'main.py' if os.path.exists('main.py') else 'main.pyc'
  os.system('python %s -dataset=avec2014.audio.cnn -task=test -model=%s' %
            (main, model[0]))
