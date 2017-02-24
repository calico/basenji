#!/usr/bin/env python
import sys

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
import seaborn as sns

def jointplot(vals1, vals2, out_pdf, alpha=0.5, square=False, cor='pearsonr', x_label=None, y_label=None):
    plt.figure()

    if cor.lower() in ['spearman','spearmanr']:
        cor_func = spearmanr
    elif cor.lower() in ['pearson','pearsonr']:
        cor_func = pearsonr
    else:
        print('Cannot recognize correlation method %s' % cor, file=sys.stderr)

    g = sns.jointplot(vals1, vals2, color='black', space=0, stat_func=cor_func, joint_kws={'alpha':alpha, 's':12})

    ax = g.ax_joint

    if square:
        vmin, vmax = scatter_lims(vals1, vals2)
        ax.set_xlim(vmin,vmax)
        ax.set_ylim(vmin,vmax)

        ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    else:
        xmin, xmax = scatter_lims(vals1)
        ax.set_xlim(xmin,xmax)
        ymin, ymax = scatter_lims(vals2)
        ax.set_ylim(ymin,ymax)

    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)

    ax.grid(True, linestyle=':')

    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, point_size=4, cor='pearsonr', square=True, x_label=None, y_label=None, title=None):
    plt.figure(figsize=(6,6))

    gold = sns.color_palette('husl',8)[1]
    ax = sns.regplot(vals1, vals2, color='black', order=poly_order, scatter_kws={'color':'black', 's':point_size, 'alpha':alpha}, line_kws={'color':gold})

    if square:
        vmin, vmax = scatter_lims(vals1, vals2)
        ax.set_xlim(vmin,vmax)
        ax.set_xlim(vmin,vmax)
    else:
        xmin, xmax = scatter_lims(vals1)
        ax.set_xlim(xmin,xmax)
        ymin, ymax = scatter_lims(vals2)
        ax.set_xlim(ymin,ymax)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        plt.title(title)

    if cor.lower() in ['spearman','spearmanr']:
        scor, _ = spearmanr(vals1, vals2)
        lim_eps = (vmax-vmin) * .02
        ax.text(vmin+lim_eps, vmax-3*lim_eps, 'Spearman R: %.3f'%scor, horizontalalignment='left', fontsize=12)
    elif cor.lower() in ['pearson','pearsonr']:
        scor, _ = pearsonr(vals1, vals2)
        lim_eps = (vmax-vmin) * .02
        ax.text(vmin+lim_eps, vmax-3*lim_eps, 'Pearson R: %.3f'%scor, horizontalalignment='left', fontsize=12)
    else:
        print('Cannot recognize correlation method %s' % cor, file=sys.stderr)

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
