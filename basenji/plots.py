# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.stats import spearmanr, pearsonr

################################################################################
# scatter plots


def jointplot(vals1,
              vals2,
              out_pdf,
              alpha=0.5,
              point_size=10,
              square=False,
              cor='pearsonr',
              x_label=None,
              y_label=None,
              figsize=(6, 6),
              sample=None,
              table=False,
              kind='scatter',
              text_means=False):

  if table:
    out_txt = '%s.txt' % out_pdf[:-4]
    out_open = open(out_txt, 'w')
    for i in range(len(vals1)):
      print(vals1[i], vals2[i], file=out_open)
    out_open.close()

  if sample is not None and sample < len(vals1):
    indexes = np.random.choice(np.arange(0, len(vals1)), sample, replace=False)
    vals1 = vals1[indexes]
    vals2 = vals2[indexes]

  if type(figsize) == tuple:
    if figsize[0] != figsize[1]:
      print('Figure size must be square', file=sys.stderr)
    figsize = figsize[0]

  plt.figure()

  if cor is None:
    cor_func = None
  elif cor.lower() in ['spearman', 'spearmanr']:
    cor_func = spearmanr
  elif cor.lower() in ['pearson', 'pearsonr']:
    cor_func = pearsonr
  else:
    cor_func = None

  if kind == 'scatter':
    joint_kws = {'alpha':alpha, 's':point_size}
  else:
    gold = sns.color_palette('husl',8)[1]
    joint_kws = {}
    joint_kws['scatter_kws'] = {'color':'black', 's':point_size, 'alpha':alpha}
    joint_kws['line_kws'] = {'color':gold}

  g = sns.jointplot(vals1, vals2,
        color='black', height=figsize,
        space=0, stat_func=cor_func,
        kind=kind, joint_kws=joint_kws)

  ax = g.ax_joint

  if square:
    vmin, vmax = scatter_lims(vals1, vals2)
    xmin = vmin
    ymin = vmin
    xmax = vmax
    ymax = vmax
    ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='black')
  else:
    xmin, xmax = scatter_lims(vals1)
    ymin, ymax = scatter_lims(vals2)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)

  if y_label is not None:
    ax.set_ylabel(y_label)
  if x_label is not None:
    ax.set_xlabel(x_label)

  if text_means:
    u1 = np.mean(vals1)
    u2 = np.mean(vals2)

    eps = .05
    text_xeps = eps*(xmax-xmin)
    test_yeps = eps*(ymax-ymin)

    # ax.text(xmax+text_xeps, ymin-test_yeps, 'mean %.3f'%u1, horizontalalignment='right', fontsize=14)
    # ax.text(xmin-text_xeps, ymax+test_yeps, 'mean %.3f'%u2, horizontalalignment='left', fontsize=14)

    ax.text(1-eps, eps, 'mean %.3f'%u1, horizontalalignment='right', transform=ax.transAxes)
    ax.text(eps, 1-eps, 'mean %.3f'%u2, verticalalignment='top', transform=ax.transAxes)

  # ax.grid(True, linestyle=':')
  # plt.tight_layout(w_pad=0, h_pad=0)

  plt.savefig(out_pdf)
  plt.close()


def regplot(vals1,
            vals2,
            out_pdf,
            poly_order=1,
            alpha=0.5,
            point_size=10,
            cor='pearsonr',
            print_sig=False,
            square=True,
            x_label=None,
            y_label=None,
            title=None,
            figsize=(6, 6),
            sample=None,
            table=False):

  if table:
    out_txt = '%s.txt' % out_pdf[:-4]
    out_open = open(out_txt, 'w')
    for i in range(len(vals1)):
      print(vals1[i], vals2[i], file=out_open)
    out_open.close()

  if sample is not None and sample < len(vals1):
    indexes = np.random.choice(np.arange(0, len(vals1)), sample, replace=False)
    vals1 = vals1[indexes]
    vals2 = vals2[indexes]

  plt.figure(figsize=figsize)

  gold = sns.color_palette('husl', 8)[1]
  ax = sns.regplot(
      vals1,
      vals2,
      color='black',
      order=poly_order,
      scatter_kws={'color': 'black',
                   's': point_size,
                   'alpha': alpha},
      line_kws={'color': gold})

  if square:
    xmin, xmax = scatter_lims(vals1, vals2)
    ymin, ymax = xmin, xmax
  else:
    xmin, xmax = scatter_lims(vals1)
    ymin, ymax = scatter_lims(vals2)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)

  if x_label is not None:
    ax.set_xlabel(x_label)
  if y_label is not None:
    ax.set_ylabel(y_label)

  if title is not None:
    plt.title(title)

  if cor is None:
    corr = None
  elif cor.lower() in ['spearman', 'spearmanr']:
    corr, csig = spearmanr(vals1, vals2)
    corr_str = 'SpearmanR: %.3f' % corr
  elif cor.lower() in ['pearson', 'pearsonr']:
    corr, csig = pearsonr(vals1, vals2)
    corr_str = 'PearsonR: %.3f' % corr
  else:
    corr = None

  if print_sig:
    if csig > .001:
      corr_str += '\n p %.3f' % csig
    else:
      corr_str += '\n p %.1e' % csig

  if corr is not None:
    xlim_eps = (xmax - xmin) * .03
    ylim_eps = (ymax - ymin) * .05

    ax.text(
        xmin + xlim_eps,
        ymax - 3 * ylim_eps,
        corr_str,
        horizontalalignment='left',
        fontsize=12)

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

  buf = .05 * (vmax - vmin)

  if vmin == 0:
    vmin -= buf / 2
  else:
    vmin -= buf
  vmax += buf

  return vmin, vmax


################################################################################
# nucleotides

# Thanks to Anshul Kundaje, Avanti Shrikumar
# https://github.com/kundajelab/deeplift/tree/master/deeplift/visualization


def plot_a(ax, base, left_edge, height, color):
  a_polygon_coords = [
      np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.2, 0.0]]),
      np.array([[1.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.8, 0.0]]),
      np.array([[0.225, 0.45], [0.775, 0.45], [0.85, 0.3], [0.15, 0.3]])
  ]
  for polygon_coords in a_polygon_coords:
    ax.add_patch(
        matplotlib.patches.Polygon(
            (np.array([1, height])[None, :] * polygon_coords + np.array(
                [left_edge, base])[None, :]),
            facecolor=color,
            edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
  ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=1.3,
          height=height,
          facecolor=color,
          edgecolor=color))
  ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=0.7 * 1.3,
          height=0.7 * height,
          facecolor='white',
          edgecolor='white'))
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 1, base],
          width=1.0,
          height=height,
          facecolor='white',
          edgecolor='white',
          fill=True))


def plot_g(ax, base, left_edge, height, color):
  ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=1.3,
          height=height,
          facecolor=color,
          edgecolor=color))
  ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=0.7 * 1.3,
          height=0.7 * height,
          facecolor='white',
          edgecolor='white'))
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 1, base],
          width=1.0,
          height=height,
          facecolor='white',
          edgecolor='white',
          fill=True))
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.825, base + 0.085 * height],
          width=0.174,
          height=0.415 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.625, base + 0.35 * height],
          width=0.374,
          height=0.15 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))


def plot_t(ax, base, left_edge, height, color):
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.4, base],
          width=0.2,
          height=height,
          facecolor=color,
          edgecolor=color,
          fill=True))
  ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge, base + 0.8 * height],
          width=1.0,
          height=0.2 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))


################################################################################
# sequences

default_colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def seqlogo(seq_scores, ax=None):
  if ax is None:
    ax = plt.gca()

  colors = ['red', 'blue', 'orange', 'green']
  plot_funcs = [plot_a, plot_c, plot_g, plot_t]

  seq_len = seq_scores.shape[0]
  seq_depth = seq_scores.shape[1]

  max_height = 0

  for li in range(seq_len):
    # sort nucleotides by score
    pos_scores = sorted([(seq_scores[li, ni], ni) for ni in range(seq_depth)])

    # maintain current height
    current_height = 0

    # for each nucleotide
    for di in range(seq_depth):
      score, ni = pos_scores[di]

      if score > 0:
        # plot nucleotide
        plot_funcs[ni](
            ax=ax,
            base=current_height,
            left_edge=li,
            height=score,
            color=colors[ni])

        # update height
        current_height += score

    # update max height
    max_height = max(max_height, current_height)

  # adjust limits
  xbuf = .005 * seq_len
  ax.set_xlim(0, seq_len + xbuf)

  ybuf = .05 * max_height
  ax.set_ylim(-ybuf, max_height + ybuf)

  # adjust line widths
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
