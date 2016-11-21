#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns

def jointplot(vals1, vals2, out_pdf):
    plt.figure()
    g = sns.jointplot(vals1, vals2, alpha=0.5, color='black', stat_func=spearmanr)
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

def scatter_lims(vals1, vals2, buffer=.05):
    vals = np.concatenate((vals1, vals2))
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = .05*(vmax-vmin)

    vmin -= buf
    vmax += buf

    return vmin, vmax
