#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
sns.set(style="ticks")

def jointplot(vals1, vals2, out_pdf):
    plt.figure()
    g = sns.jointplot(vals1, vals2, alpha=0.5, color='black', stat_func=spearmanr)
    # g = sns.jointplot(vals1, vals2, color='black', kind='reg', stat_func=spearmanr)
    ax = g.ax_joint
    vmin, vmax = scatter_lims(vals1, vals2)
    ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')
    ax.set_xlim(vmin,vmax)
    ax.set_xlabel('True')
    ax.set_ylim(vmin,vmax)
    ax.set_ylabel('Pred')
    ax.grid(True, linestyle=':')
    plt.tight_layout(w_pad=0, h_pad=0)
    plt.savefig(out_pdf)
    plt.close()


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5):
    plt.figure(figsize=(6,6))

    # g = sns.jointplot(vals1, vals2, alpha=0.5, color='black', stat_func=spearmanr)
    ax = sns.regplot(vals1, vals2, color='black', order=poly_order, scatter_kws={'s':4, 'alpha':alpha})

    vmin, vmax = scatter_lims(vals1, vals2)
    # ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    ax.set_xlim(vmin,vmax)
    ax.set_xlabel('True')
    ax.set_ylim(vmin,vmax)
    ax.set_ylabel('Pred')

    scor, _ = spearmanr(vals1, vals2)
    lim_eps = (vmax-vmin) * .02
    ax.text(vmin+lim_eps, vmax-3*lim_eps, 'Spearman R: %.3f'%scor, horizontalalignment='left', fontsize=12)

    ax.grid(True, linestyle=':')

    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def scatter_lims(vals1, vals2, buffer=.05):
    vals = np.concatenate((vals1, vals2))
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = .05*(vmax-vmin)

    if vmin == 0:
        vmin -= buf/2
    else:
        vmin -= buf
    vmax += buf

    return vmin, vmax
