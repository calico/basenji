#!/usr/bin/env python
import sys

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
import seaborn as sns

def jointplot(vals1, vals2, out_pdf, alpha=0.5, point_size=10, square=False, cor='pearsonr', x_label=None, y_label=None, figsize=(6,6), sample=None, table=False):

    if table:
        out_txt = '%s.txt' % out_pdf[:-4]
        out_open = open(out_txt, 'w')
        for i in range(len(vals1)):
            print(vals1[i], vals2[i], file=out_open)
        out_open.close()

    if sample is not None and sample < len(vals1):
        vals1 = np.random.choice(vals1, sample, replace=False)
        vals2 = np.random.choice(vals2, sample, replace=False)

    plt.figure(figsize=figsize)

    if cor is None:
        cor_func = None
    elif cor.lower() in ['spearman','spearmanr']:
        cor_func = spearmanr
    elif cor.lower() in ['pearson','pearsonr']:
        cor_func = pearsonr
    else:
        cor_func = None

    g = sns.jointplot(vals1, vals2, color='black', space=0, stat_func=cor_func, joint_kws={'alpha':alpha, 's':point_size})

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

    # ax.grid(True, linestyle=':')
    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, point_size=10, cor='pearsonr', square=True, x_label=None, y_label=None, title=None, figsize=(6,6), sample=None, table=False):

    if table:
        out_txt = '%s.txt' % out_pdf[:-4]
        out_open = open(out_txt, 'w')
        for i in range(len(vals1)):
            print(vals1[i], vals2[i], file=out_open)
        out_open.close()

    if sample is not None and sample < len(vals1):
        vals1 = np.random.choice(vals1, sample, replace=False)
        vals2 = np.random.choice(vals2, sample, replace=False)

    plt.figure(figsize=figsize)

    gold = sns.color_palette('husl',8)[1]
    ax = sns.regplot(vals1, vals2, color='black', order=poly_order, scatter_kws={'color':'black', 's':point_size, 'alpha':alpha}, line_kws={'color':gold})

    if square:
        xmin, xmax = scatter_lims(vals1, vals2)
        ymin, ymax = xmin, xmax
    else:
        xmin, xmax = scatter_lims(vals1)
        ymin, ymax = scatter_lims(vals2)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        plt.title(title)

    if cor is None:
        corr = None
    elif cor.lower() in ['spearman','spearmanr']:
        corr, _ = spearmanr(vals1, vals2)
        cor_label = 'Spearman'
    elif cor.lower() in ['pearson','pearsonr']:
        corr, _ = pearsonr(vals1, vals2)
        cor_label = 'Pearson'
    else:
        corr = None

    if corr is not None:
        xlim_eps = (xmax-xmin) * .02
        ylim_eps = (ymax-ymin) * .02

        ax.text(xmin+xlim_eps, ymax-3*ylim_eps, '%s R: %.3f'%(cor_label,corr), horizontalalignment='left', fontsize=12)

    # ax.grid(True, linestyle=':')
    sns.despine()

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
