
def get_result(res, val):
    correct = 0.0
    for i in range(len(res)):
        if res[i][2] == 1 and res[i][3] < val:
            correct += 1
        if res[i][2] == -1 and res[i][3] >= val:
            correct += 1
    return correct / len(res)


def get_best_threshold(res):
    size = len(res)
    _best_val = 0
    _best_err = 0
    for i in range(size):
        _cur_val = res[i][3]
        _cur_err = get_result(res, _cur_val)
        if _cur_err > _best_err:
            _best_err = _cur_err
            _best_val = _cur_val
    # print('Best Threshold:%.4f, Best Error:%.4f' % (_best_val, _best_err))
    return _best_err, _best_val


def parse_result(filepath):
    result = []
    with open(filepath) as fp:
        for line in fp:
            r = line.split(' ')
            result.append((r[0], r[1], int(r[2]), float(r[3])))
    return result


def get_kinface_error(filepath, val=None):
    if val is None:
        res = parse_result(filepath)
        return get_best_threshold(res)
    else:
        res = parse_result(filepath)
        return get_result(res, val)
