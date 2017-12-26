# -*- coding: utf-8 -*-
""" Using for computing kinface distance
"""
import numpy as np
import sklearn.decomposition.pca as pca


class Error():
  """ A class for different error computing
  """

  def __init__(self):
    pass

  def _parse_from_file(self, filepath):
    """ path1 path2 path3 label value
    """
    dists = []
    labels = []
    with open(filepath) as fp:
      for line in fp:
        r = line.split(' ')
        labels.append(int(r[4]))
        dists.append(float(r[5]))
    return np.array(dists), np.array(labels)

  def get_result_from_file(self, filepath, threshold=None):
    """
    """
    dists, labels = self._parse_from_file(filepath)
    if threshold is None:
      return self._find_best_threshold(dists, labels)
    else:
      return self._get_error(dists, labels, threshold)

  def get_val_result(self, x, y, labels):
    """
    """
    val_dist = self._get_cosine_dist(x, y)
    val_err, val_thed = self._find_best_threshold(val_dist, labels)
    return val_err, val_thed

  def get_test_result(self, x, y, labels, threshold):
    """
    """
    test_dist = self._get_cosine_dist(x, y)
    test_err = self._get_error(test_dist, labels, threshold)
    return test_err

  def get_all_result(self, val_x, val_y, val_labels,
                     test_x, test_y, test_labels, use_PCA=False):
    """ A pipline for processing the data
    """
    if use_PCA:
      self.pca = pca.PCA(n_components=200, svd_solver='full')
      val_x, val_y, test_x, test_y = self._pca_process(
          val_x, val_y, test_x, test_y)

    val_err, val_thed = self.get_val_result(val_x, val_y, val_labels)
    test_err, test_thed = self.get_val_result(test_x, test_y, test_labels)
    return val_err, test_thed, test_err

  def _get_cosine_dist(self, feat1, feat2):
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

  def _get_error(self, dists, labels, threshold):
    correct = 0.0
    for _i in range(len(dists)):
      if labels[_i] == 1 and dists[_i] <= threshold:
        correct += 1
      if labels[_i] == -1 and dists[_i] > threshold:
        correct += 1
    return correct / len(dists)

  def _find_best_threshold(self, dists, labels):
    """ from dists find a best value to divide dataset
    """
    _best_val = 0
    _best_err = 0
    for _dist in dists:
      _cur_err = self._get_error(dists, labels, _dist)
      if _cur_err > _best_err:
        _best_err = _cur_err
        _best_val = _dist
    return _best_err, _best_val

  def _pca_process(self, val_x, val_y, test_x, test_y):
    """ get new data via pca processing.
    """
    val_feats = np.row_stack((val_x, val_y))
    self.pca.fit(val_feats)
    val_feats = self.pca.transform(val_feats)

    test_feats = np.row_stack((test_x, test_y))
    test_feats = self.pca.transform(test_feats)

    _val_x = val_feats[:val_x.shape[0]]
    _val_y = val_feats[val_x.shape[0]:]
    _test_x = test_feats[:test_x.shape[0]]
    _test_y = test_feats[test_x.shape[0]:]

    return _val_x, _val_y, _test_x, _test_y
