
import re
import numpy as np

""" Input: a log file
    Output: {'class1':[label, logit], ...}
"""


def get_seq_list(path):
    """ For video input
        class1 3 2.53
        class2 4 4.67
        ...
    Return:
        {'class1':[label, logit], ...}
    """
    result = {}
    res_fp = open(path, 'r')
    for line in res_fp:
        r1 = re.findall('frames_flow\\\\(.*)_video', line)
        r2 = re.findall('frames\\\\(.*)_video', line)

        res = r1[0] if len(r1) else r2[0]

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        label = float(label[0])
        logit = float(logit[0])

        result[res] = [label, logit]
    return result


def get_succ_list(path):
    """ For succ video sequence
    """
    res_fp = open(path, 'r')
    res_label = {}
    res_logit = {}
    for line in res_fp:
        r1 = re.findall('frames_flow\\\\(.*)_video', line)
        r2 = re.findall('frames\\\\(.*)_video', line)

        res = r1[0] if len(r1) else r2[0]

        if res.find('frames') >= 0:
            print(res)

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        if res not in res_label:
            res_label[res] = [float(label[0])]
        else:
            res_label[res].append(float(label[0]))

        logit_f = float(logit[0])

        if res not in res_logit:
            res_logit[res] = [logit_f]
        else:
            res_logit[res].append(logit_f)

    # acquire mean
    result = {}
    for idx in res_label:
        label = np.mean(np.array(res_label[idx]))
        logit = np.mean(np.array(res_logit[idx]))
        result[idx] = [label, logit]

    return result


def get_img_list(path):
    """ For image input
        class1 2 3.56
        class1 2 4.32
        class1 2 1.23
        ...
        class2 3 2.11
        ...
    Return:
        Note: the logit will be mean of the value,
          because the label is same value in same class
        {'class1':[label, logit], ...}
    """
    res_fp = open(path, 'r')
    res_label = {}
    res_logit = {}
    for line in res_fp:
        r1 = re.findall('frames_flow\\\(.*?)_video', line)
        r2 = re.findall('frames\\\(.*?)_video', line)

        res = r1[0] if len(r1) else r2[0]

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        if res not in res_label:
            res_label[res] = [float(label[0])]
        else:
            res_label[res].append(float(label[0]))

        logit_f = float(logit[0])

        if res not in res_logit:
            res_logit[res] = [logit_f]
        else:
            res_logit[res].append(logit_f)

    # acquire mean
    result = {}
    for idx in res_label:
        label = np.mean(np.array(res_label[idx]))
        logit = np.mean(np.array(res_logit[idx]))
        result[idx] = [label, logit]
    return result
