# -*- coding: utf-8 -*-
""" Using for computing kinface distance
"""
import numpy as np
import sklearn.decomposition.pca


class Error():
    """ A class for different error computing
    """

    def __init__(self):
        pass

    def _avg_dists(self, dists, num):
        """
        """
        for _i in range(num):
            if _i == 0:
                avg = np.array(dists[0])
            else:
                avg = avg + np.array(dists[_i])
        avg = np.array(avg) / float(num)
        return avg

    def get_avg_ensemble(self, val_xs, val_ys, val_labels,
                         test_xs, test_ys, test_labels, use_PCA=False):
        """ feats1 is a list:
                [Model1_feat1, Model2_feat1, ...]
            labels is [N, ]
        """
        num_model = len(val_xs)
        dists_val = []
        dists_test = []
        for _i in range(num_model):
            if use_PCA:
                self.pca = sklearn.decomposition.pca.PCA(399)
                val_x, val_y, test_x, test_y = self._pca_process(
                    val_xs[_i], val_ys[_i], test_xs[_i], test_ys[_i])

            dist = self._get_cosine_dist(val_x, val_y)
            dists_val.append(dist)

            dist = self._get_cosine_dist(test_x, test_y)
            dists_test.append(dist)

        avg_dist_val = self._avg_dists(dists_val, num_model)
        avg_dist_test = self._avg_dists(dists_test, num_model)

        val_err, val_thed = self._find_best_threshold(avg_dist_val, val_labels)
        test_err = self._get_error(avg_dist_test, test_labels, val_thed)

        return val_err, val_thed, test_err

    def _parse_from_file(self, filepath):
        """ path1 path2 label value
        """
        dists = []
        labels = []
        with open(filepath) as fp:
            for line in fp:
                r = line.split(' ')
                labels.append(int(r[2]))
                dists.append(float(r[3]))
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
            self.pca = sklearn.decomposition.pca.PCA(399)
            val_x, val_y, test_x, test_y = self._pca_process(
                val_x, val_y, test_x, test_y)

        val_err, val_thed = self.get_val_result(val_x, val_y, val_labels)
        test_err = self.get_test_result(test_x, test_y, test_labels, val_thed)

        return val_err, val_thed, test_err

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
            dis = (np.dot(p, c)) / (norm_c * norm_p)
            dists.append(dis)
        return dists

    def _get_error(self, dists, labels, threshold):
        correct = 0.0
        for _i in range(len(dists)):
            if labels[_i] == 1 and dists[_i] >= threshold:
                correct += 1
            if labels[_i] == -1 and dists[_i] < threshold:
                correct += 1
            if labels[_i] == 0 and dists[_i] < threshold:
                correct += 1
        return correct / len(dists)

    def _find_best_threshold(self, dists, labels):
        """ from dists find a best value to divide dataset
        """
        _best_val = 0
        _best_err = 0
        for _i in range(len(dists)):
            _cur_val = dists[_i]
            _cur_err = self._get_error(dists, labels, _cur_val)
            if _cur_err > _best_err:
                _best_err = _cur_err
                _best_val = _cur_val
        return _best_err, _best_val

    def _pca_process(self, val_x, val_y, test_x, test_y):
        """ get new data via pca processing.
        """
        val_feats = np.row_stack((val_x, val_y))
        feat_mean = np.mean(val_feats, axis=0)
        eig_vec = self._pca_fit(val_feats)

        sum_data = np.row_stack((val_x, val_y, test_x, test_y))
        sum_data = sum_data - feat_mean
        pca_data = self._pca_map(sum_data, eig_vec)

        _val_max_dim = val_x.shape[0] + val_y.shape[0]
        _test_max_dim = _val_max_dim + test_x.shape[0] + test_y.shape[0]
        _val_x = pca_data[0: val_x.shape[0]]
        _val_y = pca_data[val_x.shape[0]: _val_max_dim]
        _test_x = pca_data[_val_max_dim: _val_max_dim + test_x.shape[0]]
        _test_y = pca_data[_val_max_dim + test_x.shape[0]:_test_max_dim]

        return _val_x, _val_y, _test_x, _test_y

    def _pca_fit(self, feats):
        """ find eig vector
        """
        self.pca.fit(feats)
        return self.pca.components_

    def _pca_map(self, feats, eig_vec):
        """ project feats to eig_vec
        """
        return np.dot(feats, np.transpose(eig_vec))