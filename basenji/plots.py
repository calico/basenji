# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import sys

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.stats import spearmanr, pearsonr

def jointplot(vals1, vals2, out_pdf, alpha=0.5, point_size=10, square=False, cor='pearsonr', x_label=None, y_label=None, figsize=(6,6), sample=None, table=False):

    if table:
        out_txt = '%s.txt' % out_pdf[:-4]
        out_open = open(out_txt, 'w')
        for i in range(len(vals1)):
            print(vals1[i], vals2[i], file=out_open)
        out_open.close()

    if sample is not None and sample < len(vals1):
        indexes = np.random.choice(np.arange(0,len(vals1)), sample, replace=False)
        vals1 = vals1[indexes]
        vals2 = vals2[indexes]

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


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, point_size=10, cor='pearsonr', print_sig=False, square=True, x_label=None, y_label=None, title=None, figsize=(6,6), sample=None, table=False):

    if table:
        out_txt = '%s.txt' % out_pdf[:-4]
        out_open = open(out_txt, 'w')
        for i in range(len(vals1)):
            print(vals1[i], vals2[i], file=out_open)
        out_open.close()

    if sample is not None and sample < len(vals1):
        indexes = np.random.choice(np.arange(0,len(vals1)), sample, replace=False)
        vals1 = vals1[indexes]
        vals2 = vals2[indexes]

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
        corr, csig = spearmanr(vals1, vals2)
        corr_str = 'SpearmanR: %.3f' % corr
    elif cor.lower() in ['pearson','pearsonr']:
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
        xlim_eps = (xmax-xmin) * .03
        ylim_eps = (ymax-ymin) * .05

        ax.text(xmin+xlim_eps, ymax-3*ylim_eps, corr_str, horizontalalignment='left', fontsize=12)

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


################################################################################
# Nucleotide plotting

# Thanks to Anshul Kundaje, Avanti Shrikumar
# https://github.com/kundajelab/deeplift/tree/master/deeplift/visualization

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))



################################################################################
# Sequence plotting

default_colors = {0:'red', 1:'blue', 2:'orange', 3:'green'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}

def plot_weights_given_ax(ax, array, highlight, height_padding_factor=0.1, colors=default_colors, plot_funcs=default_plot_funcs):

    # change to Lx4
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4

    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []

    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))

        positive_height_so_far = 0.0
        negative_height_so_far = 0.0

        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            # plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
            if letter[1] != 0:
                plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)

        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))

    length_padding = .005*array.shape[0]
    # ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.set_xlim(0, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, choose_subtick_frequency(array.shape[0])))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)


def plot_loss_gain(ax, seq_1hot, sat_loss_ti, sat_gain_ti, highlight={}, height_padding_factor=0.1, colors=default_colors, plot_funcs=default_plot_funcs):
    ''' plot_loss_gain

    Args
     ax (Axis)
     seq_1hot (Lx4 array): One-hot coding of a sequence.
     sat_loss_ti (L array): Minimum mutation delta across satmut length.
     sat_gain_ti (L array): Maximum mutation delta across satmut length.
    '''

    max_loss_height = 0.0
    max_gain_height = 0.0
    heights_at_positions = []
    depths_at_positions = []

    for li in range(seq_1hot.shape[0]):
        # determine letter, color
        plot_func = None
        for ni in range(4):
            if seq_1hot[li,ni] == 1:
                plot_func = plot_funcs[ni]
                color = colors[ni]

        if plot_func:
            # plot loss
            plot_func(ax=ax, base=0, left_edge=li, height=sat_loss_ti[li], color=color)
            max_loss_height = max(max_loss_height, sat_loss_ti[li])
            heights_at_positions.append(sat_loss_ti[li])

            # plot gain
            plot_func(ax=ax, base=0, left_edge=li, height=sat_gain_ti[li], color=color)
            max_gain_height = min(max_gain_height, sat_gain_ti[li])
            depths_at_positions.append(sat_gain_ti[li])

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))

    ax.set_xlim(0, seq_1hot.shape[0])
    ax.xaxis.set_ticks(np.arange(0.0, seq_1hot.shape[0]+1, choose_subtick_frequency(seq_1hot.shape[0])))
    height_padding = max(abs(max_gain_height)*(height_padding_factor),
                         abs(max_loss_height)*(height_padding_factor))
    ax.set_ylim(max_gain_height-height_padding, max_loss_height+height_padding)


def choose_subtick_frequency(seq_len):
    ''' Choose the sequence visualization subtick frequency
         as a function of the sequence length. '''
    st_freq = 1
    if seq_len > 160:
        st_freq = 20
    elif seq_len > 80:
        st_freq = 10
    elif seq_len > 40:
        st_freq = 5
    elif seq_len > 20:
        st_freq = 2
    return st_freq