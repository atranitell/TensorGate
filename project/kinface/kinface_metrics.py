
import kinface_distance
import numpy as np
import re
import sklearn.decomposition.pca as PCA
import scipy.stats as stats


def dis(p, c):
    p = np.array(p)
    c = np.array(c)
    norm_p = np.linalg.norm(p)
    norm_c = np.linalg.norm(c)
    dis = (np.dot(p, c)) / (norm_c * norm_p)
    return dis


def parse_features(filepath, outpath, feats, names):
    data = []
    with open(filepath) as fp:
        for line in fp:
            r = line.split(' ')
            data.append((r[0].split('\\')[1], r[1].split('\\')[1], int(r[2])))

    # (parent, child, label, distance)
    val = {}
    for k in range(len(data)):
        for idx, name in enumerate(names):
            name = str(name, encoding="utf-8")
            # P
            res = re.findall(data[k][0], name)
            if len(res):
                val[data[k][0]] = feats[idx]
            # C
            res = re.findall(data[k][1], name)
            if len(res):
                val[data[k][1]] = feats[idx]

    feat_1 = []
    feat_2 = []
    with open(outpath, 'w') as fw:
        for k in range(len(data)):
            dist = dis(val[data[k][0]], val[data[k][1]])
            fw.write(data[k][0] + ' ' + data[k][1] + ' ' +
                     str(data[k][2]) + ' ' + str(dist) + '\n')
            feat_1.append(val[data[k][0]])
            feat_2.append(val[data[k][1]])
    return data, feat_1, feat_2


def _rewrite_with_pca(data, feat1, feat2, outpath):
    with open(outpath, 'w') as fw:
        for k in range(len(data)):
            dist = dis(feat1[k], feat2[k])
            fw.write(data[k][0] + ' ' + data[k][1] + ' ' +
                     str(data[k][2]) + ' ' + str(dist) + '\n')


def get_all():
    for M in ['_data/fd_', '_data/fs_', '_data/md_', '_data/ms_']:
        feats = np.load(M + 'kinface2_250000_PostPool_features.npy')
        names = np.load(M + 'kinface2_250000_PostPool_names.npy')
        trn_errs = []
        val_errs = []
        for i in range(1, 6):
            parse_features(M + 'train_' + str(i) + '.txt',
                           M + 'train_' + str(i) + '_r.txt',
                           feats, names)
            parse_features(M + 'val_' + str(i) + '.txt',
                           M + 'val_' + str(i) + '_r.txt',
                           feats, names)
            train_err, val = kinface_distance.get_kinface_error(
                M + 'train_' + str(i) + '_r.txt')
            trn_errs.append(train_err)
            val_err = kinface_distance.get_kinface_error(
                M + 'val_' + str(i) + '_r.txt', val=val)
            val_errs.append(val_err)
            print('On validation %s: %.4f' %
                  (M + 'val_' + str(i) + '.txt', val_err))
            break
        break
        for idx in range(5):
            print(trn_errs[idx])
        for idx in range(5):
            print(val_errs[idx])
        print()


def parse_data_from_file(filename):
    with open(filename, 'r') as fp:
        features = []
        for line in fp:
            r = line.split(' ')
            if len(r):
                feat = []
                for i in range(len(r) - 1):
                    feat.append(float(r[i]))
            features.append(feat)
        return features


def compute(use_PCA=False):
    pca = PCA.PCA(n_components=399)
    for M in ['_data/fd_', '_data/fs_', '_data/md_', '_data/ms_']:

        # feats = np.load(M + 'kinface2_250000_PostPool_features.npy')
        feats = parse_data_from_file(M + 'kinface2_vgg_fc6_features.txt')
        names = np.load(M + 'kinface2_250000_PostPool_names.npy')

        mean_trn = 0
        mean_val = 0
        for i in range(1, 6):
            trn_v, trn_f1, trn_f2 = parse_features(
                M + 'train_' + str(i) + '.txt',
                M + 'train_' + str(i) + '_r.txt',
                feats, names)
            val_v, val_f1, val_f2 = parse_features(
                M + 'val_' + str(i) + '.txt',
                M + 'val_' + str(i) + '_r.txt',
                feats, names)

            if use_PCA:
                trn = np.array(trn_f1 + trn_f2)
                trn_mean = np.mean(trn, axis=0)
                pca.fit(trn)
                eig_vec = pca.components_
                dataset = np.array(trn_f1 + trn_f2 + val_f1 + val_f2)
                dataset = np.dot((dataset - trn_mean), np.transpose(eig_vec))
                trn_f1 = dataset[0:400]
                trn_f2 = dataset[400:800]
                val_f1 = dataset[800:900]
                val_f2 = dataset[900:1000]
                _rewrite_with_pca(trn_v, trn_f1, trn_f2, M +
                                  'train_' + str(i) + '_r.txt')
                _rewrite_with_pca(val_v, val_f1, val_f2, M +
                                  'val_' + str(i) + '_r.txt')

            train_err, val = kinface_distance.get_kinface_error(
                M + 'train_' + str(i) + '_r.txt')
            val_err = kinface_distance.get_kinface_error(
                M + 'val_' + str(i) + '_r.txt', val=val)
            print('On validation %s: %.4f' %
                  (M + 'val_' + str(i) + '.txt', val_err))
            mean_trn += train_err
            mean_val += val_err

        print('mean train: %.4f, mean val: %.4f ' %
              (mean_trn / 5., mean_val / 5.))
        print()


def compute_all(use_PCA=False, save_feature=True):
    pca = PCA.PCA(n_components=399)
    # concat all database
    for idx, M in enumerate(['_data/fd_', '_data/fs_', '_data/md_', '_data/ms_']):
        feat = np.load(M + 'kinface2_250000_PostPool_features.npy')
        # feat = parse_data_from_file(M + 'kinface2_vgg_fc6_features.txt')
        # feat = np.load(M + 'kinface2_80000_PostPool_features.npy')
        name = np.load(M + 'kinface2_250000_PostPool_names.npy')
        if idx == 0:
            feats = feat
            names = name
        else:
            feats = np.row_stack([feats, feat])
            names = np.append(names, name)

    # (2000, N)
    print(feats.shape)
    # (2000, 1)
    print(names.shape)

    trn_list = []
    val_list = []
    for i in range(1, 6):
        trn_v, trn_f1, trn_f2 = parse_features(
            '_data/train_' + str(i) + '.txt',
            '_data/train_' + str(i) + '_r.txt',
            feats, names)
        val_v, val_f1, val_f2 = parse_features(
            '_data/val_' + str(i) + '.txt',
            '_data/val_' + str(i) + '_r.txt',
            feats, names)

        if save_feature:
            trn_label = []
            for k in range(len(trn_v)):
                trn_label.append(int(trn_v[k][2]))
            trn_label = np.reshape(np.array(trn_label), (len(trn_v), 1))

            val_label = []
            for k in range(len(val_v)):
                val_label.append(int(val_v[k][2]))
            val_label = np.reshape(np.array(val_label), (len(val_v), 1))

            np.save('_data/train_' + str(i) + '_1.npy', trn_f1)
            np.save('_data/train_' + str(i) + '_2.npy', trn_f2)
            np.save('_data/train_' + str(i) + '_label.npy', trn_label)

            np.save('_data/val_' + str(i) + '_1.npy', val_f1)
            np.save('_data/val_' + str(i) + '_2.npy', val_f2)
            np.save('_data/val_' + str(i) + '_label.npy', val_label)

        if use_PCA:
            trn = np.array(trn_f1 + trn_f2)
            trn_mean = np.mean(trn, axis=0)
            pca.fit(trn)
            eig_vec = pca.components_
            dataset = np.array(trn_f1 + trn_f2 + val_f1 + val_f2)
            dataset = np.dot((dataset - trn_mean), np.transpose(eig_vec))
            trn_f1 = dataset[0:1600]
            trn_f2 = dataset[1600:3200]
            val_f1 = dataset[3200:3600]
            val_f2 = dataset[3600:4000]
            _rewrite_with_pca(
                trn_v, trn_f1, trn_f2, '_data/train_' + str(i) + '_r.txt')
            _rewrite_with_pca(
                val_v, val_f1, val_f2, '_data/val_' + str(i) + '_r.txt')

        train_err, val = kinface_distance.get_kinface_error(
            '_data/train_' + str(i) + '_r.txt')
        val_err = kinface_distance.get_kinface_error(
            '_data/val_' + str(i) + '_r.txt', val=val)
        print('On validation %s: %.4f' %
              ('val_' + str(i) + '.txt', val_err))
        trn_list.append(train_err)
        val_list.append(val_err)
        print()

    print(trn_list)
    print(val_list)


# compute()
compute_all()
