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
"""Using for computing cosine similarity"""

import numpy as np


def compute_cosine_similarity(feat1, feat2):
  """Feat [N, M]

  Shape:
      N: number of feat
      M: dim of feature

  Return:
      A [N] list represent distance between feat1 and feat2
  """
  dists = []
  for _i in range(feat1.shape[0]):
    p = np.array(feat1[_i])
    c = np.array(feat2[_i])
    norm_p = np.linalg.norm(p)
    norm_c = np.linalg.norm(c)
    dis = 1 - (np.dot(p.T, c)) / (norm_c * norm_p)
    dists.append(dis)
  return dists


def compute_error(dists, labels, threshold):
  """Compute similarity with threshold"""
  correct = 0.0
  for _i in range(len(dists)):
    if labels[_i] == 1 and dists[_i] <= threshold:
      correct += 1
    if labels[_i] == -1 and dists[_i] > threshold:
      correct += 1
  return correct / len(dists)


def compute_best_error(dists, labels):
  """distinguish with compute error"""
  _best_thred, _best_err = 0, 0
  for _dist in dists:
    _cur_err = compute_error(dists, labels, _dist)
    if _cur_err > _best_err:
      _best_err = _cur_err
      _best_thred = _dist
  return _best_err, _best_thred


def get_result(feat_x, feat_y, labels, thred=None):
  """If thred is None, find a best margin, else computing error with thred"""
  dists = compute_cosine_similarity(feat_x, feat_y)
  if thred is None:
    err, thed = compute_best_error(dists, labels)
  else:
    err, thred = compute_error(dists, labels, thred), thred
  return err, thed


def get_all_result(v_x, v_y, v_l, t_x, t_y, t_l, use_val_thred=False):
  # compute thred in validation set
  val_err, val_thed = get_result(v_x, v_y, v_l)
  # compute error in test set
  if use_val_thred:
    test_err, test_thed = get_result(t_x, t_y, t_l, val_thed)
  else:
    test_err, test_thed = get_result(t_x, t_y, t_l)
  return val_err, test_thed, test_err


def get_result_from_file(filepath, threshold=None):
  dists, labels = [], []
  with open(filepath) as fp:
    for line in fp:
      r = line.split(' ')
      labels.append(int(r[4]))
      dists.append(float(r[5]))
  dists, labels = np.array(dists), np.array(labels)
  if threshold is None:
    return compute_best_error(dists, labels)
  else:
    return compute_error(dists, labels, threshold)
