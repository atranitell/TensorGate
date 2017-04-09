
import matplotlib.pyplot as plt
import re


def get_succ_list(path):
    """ For succ video sequence
    """
    res_fp = open(path, 'r')
    res_label = {}
    res_logit = {}
    for line in res_fp:
        r1 = re.findall('frames\\\(.*)_video', line)
        r2 = re.findall('frames_flow\\\(.*)_video', line)

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
    return res_label, res_logit

res_label, res_logit = get_succ_list('analysis/avec2014/26001.txt')
res_label1, res_logit1 = get_succ_list('analysis/avec2014/15001.txt')

cc = []
for i in res_label:
    c = len(res_label[i])
    cc.append((c, i))

cc = sorted(cc)
for i in cc:
    print(i)

for k in res_label:
    count = [i for i in range(len(res_label[k]))]

    plt.plot(count, res_label[k], 'r', alpha=0.6)
    plt.plot(count, res_logit[k], 'b', alpha=0.6)
    plt.plot(count, res_logit1[k], 'g', alpha=0.6)

    plt.grid()
    # plt.xlim((0, info[iter_tag][-1]))
    plt.xlabel('iter')
    plt.ylabel('')
    name = str(int(res_label[k][0])).zfill(2) + '_' + k
    plt.title(name)
    # plt.show()
    plt.savefig(name)
    plt.close()
