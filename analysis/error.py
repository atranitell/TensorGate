
import math


def get_mae_rmse(res):
    """
    Input: a dict
        {'class1':[label value],
         'class2':[label value]}
    """
    mae = 0.0
    rmse = 0.0
    for idx in res:
        mae += abs(res[idx][0] - res[idx][1])
        rmse += math.pow(res[idx][0] - res[idx][1], 2)
    mae = mae / len(res)
    rmse = math.sqrt(rmse / len(res))
    return mae, rmse, len(res)


def get_mae(res):
    mae = 0.0
    for idx in res:
        mae += abs(res[idx][0] - res[idx][1])
    mae = mae / len(res)
    return mae, len(res)


def get_rmse(res):
    rmse = 0.0
    for idx in res:
        rmse += math.pow(res[idx][0] - res[idx][1], 2)
    rmse = math.sqrt(rmse / len(res))
    return rmse, len(res)
