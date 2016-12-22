#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns

def jointplot(vals1, vals2, out_pdf, alpha=0.5, x_label=None, y_label=None):
    plt.figure()

    g = sns.jointplot(vals1, vals2, alpha=alpha, color='black', stat_func=spearmanr)

    ax = g.ax_joint
    vmin, vmax = scatter_lims(vals1, vals2)
    ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    ax.set_xlim(vmin,vmax)
    ax.set_xlabel(x_label)
    ax.set_ylim(vmin,vmax)
    ax.set_ylabel(y_label)

    ax.grid(True, linestyle=':')

    plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, x_label=None, y_label=None):
    plt.figure(figsize=(6,6))

    # g = sns.jointplot(vals1, vals2, alpha=0.5, color='black', stat_func=spearmanr)
    gold = sns.color_palette('husl',8)[1]
    ax = sns.regplot(vals1, vals2, color='black', order=poly_order, scatter_kws={'color':'black', 's':4, 'alpha':alpha}, line_kws={'color':gold})

    vmin, vmax = scatter_lims(vals1, vals2)
    # ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    ax.set_xlim(vmin,vmax)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylim(vmin,vmax)
    if y_label is not None:
        ax.set_ylabel(y_label)

    scor, _ = spearmanr(vals1, vals2)
    lim_eps = (vmax-vmin) * .02
    ax.text(vmin+lim_eps, vmax-3*lim_eps, 'Spearman R: %.3f'%scor, horizontalalignment='left', fontsize=12)

    ax.grid(True, linestyle=':')

    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def scatter_lims(vals1, vals2=None, buffer=.05):
    if vals2 is not None:
        vals = np.concatenate((vals1, vals2))
    else:
        vals = vals1
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = .05*(vmax-vmin)

    if vmin == 0:
        vmin -= buf/2
    else:
        vmin -= buf
    vmax += buf

    return vmin, vmax
