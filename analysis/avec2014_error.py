
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