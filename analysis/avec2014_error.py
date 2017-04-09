
import os
from analysis import avec2014_textfile
from analysis import error


def get_accurate_from_file(path, data_type):

    if data_type == 'seq':
        res = error.get_mae_rmse(avec2014_textfile.get_seq_list(path))
    elif data_type == 'img':
        res = error.get_mae_rmse(avec2014_textfile.get_img_list(path))
    elif data_type == 'succ':
        res = error.get_mae_rmse(avec2014_textfile.get_succ_list(path))

    mae, rmse, count = res
    return mae, rmse


def get_mae_rmse(res):
    return error.get_mae_rmse(res)


def get_all_res_in_fold(path, data_type):
    print(path)

    min_mae = 10000
    min_mae_iter = ''
    min_rmse = 10000
    min_rmse_iter = ''
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)

        if data_type == 'seq':
            res = get_mae_rmse(avec2014_textfile.get_seq_list(filepath))
        elif data_type == 'img':
            res = get_mae_rmse(avec2014_textfile.get_img_list(filepath))
        elif data_type == 'succ':
            res = get_mae_rmse(avec2014_textfile.get_succ_list(filepath))

        if res[0] < min_mae:
            min_mae = res[0]
            min_mae_iter = filename

        if res[1] < min_rmse:
            min_rmse = res[1]
            min_rmse_iter = filename

        print('(%s, %.4f, %.4f, %d)' % (filename, res[0], res[1], res[2]))

    print('min_mae: %.4f in %s' % (min_mae, min_mae_iter))
    print('min_rmse: %.4f in %s' % (min_rmse, min_rmse_iter))
