import re
import numpy as np


def get_accurate_from_file(path):

    res_label = {}
    res_logit = {}
    res_fp = open(path, 'r')

    for line in res_fp:
        r1 = re.findall('/(.*?)_Freeform_video', line)
        r2 = re.findall('/(.*?)_Northwind_video', line)

        if r1:
            res = r1[0]
        elif r2:
            res = r2[0]
        else:
            r1 = re.findall('\\\(.*?)_Freeform_video', line)
            r2 = re.findall('\\\(.*?)_Northwind_video', line)
            if r1:
                res = r1[0]
            elif r2:
                res = r2[0]
            else:
                continue

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        if res not in res_label:
            res_label[res] = [float(label[0])]
        else:
            res_label[res].append(float(label[0]))

        if res not in res_logit:
            res_logit[res] = [float(logit[0])]
        else:
            res_logit[res].append(float(logit[0]))

    mae = []
    rmse = []
    for idx in res_label:
        a = np.mean(np.array(res_label[idx]))
        b = np.mean(np.array(res_logit[idx]))
        mae.append(np.abs(a - b))
        rmse.append(np.square(a - b))

    err_mae = np.mean(mae)
    err_rmse = np.sqrt(np.mean(rmse))

    return err_mae, err_rmse

# print(get_accurate_from_file('200.txt'))
