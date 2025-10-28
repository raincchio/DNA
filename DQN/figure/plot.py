#!/home/chenxing/miniconda3/envs/Plot/bin/python3
import os

import matplotlib.pyplot as plt
import numpy as np

from xingplot.plot import XPlotter

xplot = XPlotter()
DOMAINS = ['BeamRiderNoFrameskip-v0','SpaceInvadersNoFrameskip-v0', 'AsterixNoFrameskip-v0', 'SeaquestNoFrameskip-v4', 'DemonAttackNoFrameskip-v0',]

DOMAINS = ['DemonAttackNoFrameskip-v0', ]

path = '/vepfs-dev/xing/workspace/DNA/experiments'
algos = [
    # 'DQN_muon',
    'DQN_redo',
    "DQN_redo_wob",
    'test_eps_dqn_redo',
# 'redo_test',
]
metric = "eval_reward"

COLORS = ['#77AC30', '#A56DB0', "#F0C04A",'#FF66B2', '#DE6C3A', '#2988C7', '#0000FF','#FF66B2']
# COLORS = ["#ccb974", '#c44e52', '#8172b2', '#55a868', '#4c72b0', '#0000FF']
MARKERS = ['o', '*', 's', '^', 'x','D']

# metric = "q_value"
# Plot line chart

# Plot lines

for idd, domain in enumerate(DOMAINS):
    print(idd, domain)
    # max_len = 0
    # max_y_value = -1e6
    # min_y_value = 1e6
    for algo in algos:
        algo_path = os.path.join(path, algo)  # can  get multiple metrics result
        res_ = xplot.get_result(algo_path, domain=domain, metric=metric)
        res = xplot.smooth_result(res_,10)
        if len(res)==0:
            # print(idd, domain, algo, 'zero data')
            continue
        mean = np.mean(res, axis=1).round(2)
        std = np.std(res, axis=1).round(2)
        # try:
        #     print(f'{algo}: {mean[2000]}±{std[2000]} {mean[4000]}±{std[4000]} {mean[6000]}±{std[6000]} {mean[8000]}±{std[8000]} {mean[-1]}±{std[-1]}')
        # except Exception as e:
        #     print(f'{algo}: {mean[200]}±{std[200]} {mean[400]}±{std[400]}')
        x_vals = np.arange(len(mean))  # x axis item interval

        color = COLORS[algos.index(algo)%8]
        marker = MARKERS[algos.index(algo) % 6]
        marker_num = 8
        makerevery = x_vals[-1] // marker_num

        _algo_label = algo
        plt.plot(x_vals, mean, label=_algo_label, marker=marker, markevery=makerevery, color=color, markersize=4,linewidth=1)
        plt.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.3)
        # print(idd, domain, algo, mean[-1], std[-1], len(mean))

    # Plot misc
    plt.ylabel('episode Reward')
    plt.xlabel('steps(50K)/1e7')

    # x_tick_interval = max_len//5  # just want five ticks, let it be max_len//5
    # plt.xticks([0,200,400,600,800,1000], ['0', '0.2', '0.4', '0.6', '0.8','1'])
    # plt.xlim(0, x_ticks[-1]+1)
    # plt.ylim(np.floor(min_y_value), np.ceil(max_y_value))
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    plt.grid(True, linestyle='-', alpha=0.5)
    if idd%1==0:
        lgd = plt.legend(loc='upper left', fancybox=False,
                         framealpha=1, edgecolor='black', fontsize=6)
        lgd.get_frame().set_alpha(None)
        lgd.get_frame().set_facecolor((0, 0, 0, 0))
    plt.tight_layout()
    plt.title('('+chr(65+idd)+') '+domain)

    # plt.show()
    plt.savefig('res/{}-{}.png'.format(idd,domain), bbox_inches='tight',  dpi=300)
    print('{}-{}-res.png'.format(idd,domain))
    plt.clf()
