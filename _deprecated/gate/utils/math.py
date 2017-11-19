""" A series of user-defined ops
    All input should be numpy format
"""


def find_max(logits1, logits2):
    """ find the position with max value in (N, ) vector
        argmax(x, y) = sigma_{i=0}^{N}x_i*y_i
    """
    max_val = -9999
    max_pos = 0
    dim = logits1.shape[0]
    for _i in range(dim):
        v = logits1[_i] * logits2[_i]
        if v > max_val:
            max_val = v
            max_pos = _i
    return max_val, max_pos
