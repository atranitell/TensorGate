
import os
import json
import avec2014_error


def _parse_json(path):
    with open(path, 'r') as fp:
        json_file = json.load(fp)
    for val in json_file['list']:
        val['path'] = os.path.join(
            os.path.join(os.path.dirname(path), val['path']))
    return json_file


def _get_ensemble_avg(ensemble, res_list):
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


def _print_name_mae_rmse(name_list, mae_list, rmse_list):
    print()
    for i in name_list:
        print('[FILE] ', i)
    for idx in range(len(mae_list)):
        print('[RESULT] mae: %.4f, rmse: %.4f' %
              (mae_list[idx], rmse_list[idx]))


def get_ensemble_value(json_file_path):

    models = _parse_json(json_file_path)

    res = {}
    res_list = []
    for idx, val in enumerate(models['list']):
        # print(val['path'])
        if val['type'] == 'seq':
            res_dict = avec2014_error.get_seq_list(val['path'])
        elif val['type'] == 'img':
            res_dict = avec2014_error.get_img_list(val['path'])
        elif val['type'] == 'succ':
            res_dict = avec2014_error.get_succ_list(val['path'])

        mae, rmse, count = avec2014_error.get_mae_rmse(res_dict)
        res[val['model']] = [val['ensemble'], mae, rmse, str(count)]

        if val['ensemble'] is True:
            res_list.append((res_dict))

    print('%-20s %5s     %5s    %7s   %5s' %
          ('model name:', 'mae', 'rmse', 'ensemble', 'PP'))

    for line in sorted(res):
        if res[line][2] >= 10:
            print('%-20s  %-.3f     %-.3f    %-7s    %-7s' %
                  (line, res[line][1], res[line][2], str(res[line][0]), res[line][3]))
        else:
            print('%-20s  %-.3f     %-.3f     %-7s    %-7s' %
                  (line, res[line][1], res[line][2], str(res[line][0]), res[line][3]))
    # ensemble
    ensemble = {}
    _get_ensemble_avg(ensemble, res_list)
    print('\n%-30s  %-.3f     %-.3f' %
          ('ensemble_avg', ensemble['avg_res'][0],  ensemble['avg_res'][1]))
