# -*- coding: utf-8 -*-
""" Using for computing cosine similarity
"""
import numpy as np
import sklearn.decomposition.pca as pca


def compute_cosine_similarity(feat1, feat2):
  """ feat [N, M]
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
  """ compute similarity with threshold
  """
  correct = 0.0
  for _i in range(len(dists)):
    if labels[_i] == 1 and dists[_i] <= threshold:
      correct += 1
    if labels[_i] == -1 and dists[_i] > threshold:
      correct += 1
  return correct / len(dists)


def compute_best_error(dists, labels):
  """ distinguish with compute error
  """
  _best_thred, _best_err = 0, 0
  for _dist in dists:
    _cur_err = compute_error(dists, labels, _dist)
    if _cur_err > _best_err:
      _best_err = _cur_err
      _best_thred = _dist
  return _best_err, _best_thred


def get_result(feat_x, feat_y, labels, thred=None):
  """ if thred is None, find a best margin
   else directly computing error with thred
  """
  dists = compute_cosine_similarity(feat_x, feat_y)
  if thred is None:
    err, thed = compute_best_error(dists, labels)
  else:
    err, thred = compute_error(dists, labels, thred), thred
  return err, thed


def get_all_result(v_x, v_y, v_l, t_x, t_y, t_l,
                   use_PCA=False, use_val_thred=False):
  # using PCA to reduce dim
  if use_PCA:
    v_x, v_y, t_x, t_y = transform_PCA(v_x, v_y, t_x, t_y)

  # compute thred in validation set
  val_err, val_thed = get_result(v_x, v_y, v_l)

  # compute error in test set
  if use_val_thred:
    test_err, test_thed = get_result(t_x, t_y, t_l, val_thed)
  else:
    test_err, test_thed = get_result(t_x, t_y, t_l)

  return val_err, test_thed, test_err


def transform_PCA(val_x, val_y, test_x, test_y):
  """ get new data via pca processing.
  """
  val_feats = np.row_stack((val_x, val_y))
  test_feats = np.row_stack((test_x, test_y))

  _pca = pca.PCA(n_components=200, svd_solver='full')
  _pca.fit(val_feats)

  val_feats = _pca.transform(val_feats)
  test_feats = _pca.transform(test_feats)

  _val_x = val_feats[:val_x.shape[0]]
  _val_y = val_feats[val_x.shape[0]:]
  _test_x = test_feats[:test_x.shape[0]]
  _test_y = test_feats[test_x.shape[0]:]

  return _val_x, _val_y, _test_x, _test_y


def get_result_from_file(filepath, threshold=None):
  """
  """
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
