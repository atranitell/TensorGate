
import os
import sys

sys.path.append('.')

import json

from analysis import avec2014_error
from analysis import avec2014_textfile


def parse_json(path):
    with open(path, 'r') as fp:
        json_file = json.load(fp)
    for val in json_file['list']:
        val['path'] = os.path.join(
            os.path.join(os.path.dirname(path), val['path']))
    return json_file


def get_ensemble_avg(ensemble, res_list):
    ensemble['avg'] = {}
    for idx, val in enumerate(res_list):
        # acquire first model
        if idx == 0:
            ensemble['avg'] = val
            continue

        for key in val:
            if ensemble['avg'][key][0] != val[key][0]:
                raise ValueError('Error!')
            ensemble['avg'][key][1] += val[key][1]

    for key in ensemble['avg']:
        ensemble['avg'][key][1] /= float(len(res_list))

    ensemble['avg_res'] = avec2014_error.get_mae_rmse(ensemble['avg'])


def get_best_ensemble_value2(json_file_path):

    models = parse_json(json_file_path)

    res_list = []
    for idx, val in enumerate(models['list']):
        # print(val['path'])
        if val['type'] == 'seq':
            res_dict = avec2014_textfile.get_seq_list(val['path'])
        elif val['type'] == 'img':
            res_dict = avec2014_textfile.get_img_list(val['path'])
        elif val['type'] == 'succ':
            res_dict = avec2014_textfile.get_succ_list(val['path'])

        mae, rmse, count = avec2014_error.get_mae_rmse(res_dict)
        res_list.append([val['ensemble'], mae, rmse, val[
                        'desc'], count, res_dict, val['model']])

    res_best = {}
    for res in sorted(res_list):
        if res[3] not in res_best:
            res_best[res[3]] = []
        res_best[res[3]].append(res)

    # kind_of_model = len(res_best)
    ensemble_list = [1, 2, 3]
    name = [1, 2, 3]

    min_mae_name = []
    min_rmse_name = []
    min_mae, min_mae_rmse = 100, 100
    min_rmse, min_rmse_mae = 100, 100

    for i1 in res_best['avec2014']:
        ensemble_list[0] = i1[5]
        name[0] = i1[6]
        for i3 in res_best['avec2014_16']:
            ensemble_list[1] = i3[5]
            name[1] = i3[6]
            for i4 in res_best['avec2014_flow_16']:
                ensemble_list[2] = i4[5]
                name[2] = i4[6]
                ensemble = {}
                # print(ensemble_list)
                get_ensemble_avg(ensemble, ensemble_list)
                print('%s  %-.3f  %-.3f' %
                      ('ensemble_avg', ensemble['avg_res'][0],  ensemble['avg_res'][1]))
                if min_mae > ensemble['avg_res'][0]:
                    min_mae = ensemble['avg_res'][0]
                    min_mae_rmse = ensemble['avg_res'][1]
                    min_mae_name = name.copy()

                if min_rmse > ensemble['avg_res'][1]:
                    min_rmse = ensemble['avg_res'][1]
                    min_rmse_mae = ensemble['avg_res'][0]
                    min_rmse_name = name.copy()

    print_name_mae_rmse(min_mae_name, [min_mae], [min_mae_rmse])
    print_name_mae_rmse(min_rmse_name, [min_rmse_mae], [min_rmse])


def get_best_ensemble_value(json_file_path):

    models = parse_json(json_file_path)

    res_list = []
    for idx, val in enumerate(models['list']):
        # print(val['path'])
        if val['type'] == 'seq':
            res_dict = avec2014_textfile.get_seq_list(val['path'])
        elif val['type'] == 'img':
            res_dict = avec2014_textfile.get_img_list(val['path'])
        elif val['type'] == 'succ':
            res_dict = avec2014_textfile.get_succ_list(val['path'])

        mae, rmse, count = avec2014_error.get_mae_rmse(res_dict)
        res_list.append([val['ensemble'], mae, rmse, val[
                        'desc'], count, res_dict, val['model']])

    res_best = {}
    for res in sorted(res_list):
        if res[3] not in res_best:
            res_best[res[3]] = []
        res_best[res[3]].append(res)

    # kind_of_model = len(res_best)
    ensemble_list = [1, 2, 3, 4]
    name = [1, 2, 3, 4]

    min_mae_name = []
    min_rmse_name = []
    min_mae, min_mae_rmse = 100, 100
    min_rmse, min_rmse_mae = 100, 100

    for i1 in res_best['avec2014']:
        ensemble_list[0] = i1[5]
        name[0] = i1[6]
        for i2 in res_best['avec2014_flow_16_succ']:
            ensemble_list[1] = i2[5]
            name[1] = i2[6]
            for i3 in res_best['avec2014_16']:
                ensemble_list[2] = i3[5]
                name[2] = i3[6]
                for i4 in res_best['avec2014_flow_16']:
                    ensemble_list[3] = i4[5]
                    name[3] = i4[6]
                    ensemble = {}
                    # print(ensemble_list)
                    get_ensemble_avg(ensemble, ensemble_list)
                    # print('%s  %-.3f  %-.3f' %
                    #       ('ensemble_avg', ensemble['avg_res'][0],  ensemble['avg_res'][1]))
                    if min_mae > ensemble['avg_res'][0]:
                        min_mae = ensemble['avg_res'][0]
                        min_mae_rmse = ensemble['avg_res'][1]
                        min_mae_name = name.copy()

                    if min_rmse > ensemble['avg_res'][1]:
                        min_rmse = ensemble['avg_res'][1]
                        min_rmse_mae = ensemble['avg_res'][0]
                        min_rmse_name = name.copy()

    print_name_mae_rmse(min_mae_name, [min_mae], [min_mae_rmse])
    print_name_mae_rmse(min_rmse_name, [min_rmse_mae], [min_rmse])


def print_name_mae_rmse(name_list, mae_list, rmse_list):
    print()
    for i in name_list:
        print('[FILE] ', i)
    for idx in range(len(mae_list)):
        print('[RESULT] mae: %.4f, rmse: %.4f' %
              (mae_list[idx], rmse_list[idx]))


def get_ensemble_value(json_file_path):

    models = parse_json(json_file_path)

    res = {}
    res_list = []
    for idx, val in enumerate(models['list']):
        # print(val['path'])
        if val['type'] == 'seq':
            res_dict = avec2014_textfile.get_seq_list(val['path'])
        elif val['type'] == 'img':
            res_dict = avec2014_textfile.get_img_list(val['path'])
        elif val['type'] == 'succ':
            res_dict = avec2014_textfile.get_succ_list(val['path'])

        mae, rmse, _ = avec2014_error.get_mae_rmse(res_dict)
        res[val['model']] = [val['ensemble'], mae, rmse, val['desc']]

        if val['ensemble'] is True:
            res_list.append((res_dict))

    print('%-60s %5s     %5s    %7s    %7s' %
          ('model name:', 'mae', 'rmse', 'ensemble', 'desc'))

    for line in sorted(res):
        if res[line][2] >= 10:
            print('%-60s  %-.3f     %-.3f    %-7s    %-7s' %
                  (line, res[line][1], res[line][2], str(res[line][0]), res[line][3]))
        else:
            print('%-60s  %-.3f     %-.3f     %-7s    %-7s' %
                  (line, res[line][1], res[line][2], str(res[line][0]), res[line][3]))
    # ensemble
    ensemble = {}
    get_ensemble_avg(ensemble, res_list)
    print('\n%-30s  %-.3f     %-.3f' %
          ('ensemble_avg', ensemble['avg_res'][0],  ensemble['avg_res'][1]))

# get_ensemble_value('list_old.json')

get_best_ensemble_value('analysis/avec2014/list.json')

# get_all_res_in_fold('test/avec2014_16f_train_201703220924', 'seq')
