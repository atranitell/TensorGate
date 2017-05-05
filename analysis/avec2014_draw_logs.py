
import json
import avec2014_draw_log

import matplotlib.pyplot as plt

filename = 'avec2014/avec2014_gap_log.json'
with open(filename) as fp:
    infos = json.load(fp)

show_list = []
for idx in infos['logs']:
    if idx['show'] is True:
        show_list.append(idx)

for idx in show_list:
    if infos['phase'] == 'train':
        idx['data'] = avec2014_draw_log.parse_log_train(idx['path'])
        ylim = (0, 12)
    elif infos['phase'] == 'test':
        idx['data'] = avec2014_draw_log.parse_log_test(idx['path'])
        ylim = (8, 14)
    else:
        raise ValueError('Unkonwn type.')

legends = ()
for idx in show_list:
    data = idx['data']
    ylabel = infos['ylabel']
    xlabel = infos['xlabel']
    plt.plot(data[xlabel], avec2014_draw_log.smooth(data[ylabel], 5), alpha=1)
    legends += (idx['legend'],)

plt.grid()
plt.legend(legends)
plt.xlim((0, data['iter'][-1]))
plt.ylim(ylim)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(infos['title'])
plt.show()