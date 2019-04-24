import numpy as np
from os import path
import cPickle
import sys, os
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import ConfigParser

from plotting_utils import *
from RL_utils import *

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 32}
plt.rc('font', **font)

plt.rcParams['text.latex.preamble'] = [r'\boldmath']

FONT_SIZE = 22
LEGEND_FONT_SIZE = 18

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': FONT_SIZE,
         'ytick.labelsize': FONT_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)




def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--KL_results_dir', type=str, required=True, help="")

    parser.add_argument('--entropy_results_dir', type=str, required=True, help="")

    parser.add_argument('--output_results_dir', type=str, default=None, required=False, help="")

    parser.add_argument('--config_file', type=str, required=False, help="")

    parser.add_argument('--xmax', type=int, required=False, default = 1500, help="")

    args = parser.parse_args()
    return args


def plot_overlaid_rewards(KL_rmeans = None,
                       KL_rstds = None,
                       entropy_rmeans = None,
                       entropy_rstds = None,
                       fig_path='./figs/train_reward',
                       xlabel=r'$\mathrm{Training~episodes}$',
                       ylabel=r'$\mathrm{Scaled~reward}$',
                       xlim = None,
                       ylim = None,
                       title_str = None,
                       x=None):
    if x is None:
        x = range(KL_rmeans.shape[0])


    fig = plt.figure()
    KL_color = 'green'
    entropy_color = 'blue'

    # plot KL and its std
    plt.plot(x, KL_rmeans, linewidth=2, color = KL_color)
    plt.fill_between(
        x, KL_rmeans - KL_rstds, KL_rmeans + KL_rstds, edgecolor='none', alpha=0.4, color = KL_color)

    # plot entropy and its std
    plt.plot(x, entropy_rmeans, linewidth=2, color = entropy_color)
    plt.fill_between(
        x, entropy_rmeans - entropy_rstds, entropy_rmeans + entropy_rstds, edgecolor='none', alpha=0.4, color = entropy_color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    if title_str:
        plt.title(title_str)

    #plt.legend(['KL', 'entropy'], loc = 'center right')
    #plt.legend(['KL', 'entropy'], loc = 'best')
    #plt.legend(['KL', 'entropy'], loc = 'lower right')
    plt.legend([r'$\mathrm{KL}$', r'$\mathrm{entropy}$'], loc = 'center')
    fig.savefig(fig_path)
    plt.close()


def plot_train_rewards(rmeans,
                       rstds,
                       fig_path='./figs/train_reward',
                       save_reward_path = None,
                       xlabel='Training episodes',
                       ylabel='Total rewards',
                       x=None,
                       color = 'blue'):
    if x is None:
        x = range(rmeans.shape[0])
    fig = plt.figure()
    plt.plot(x, rmeans, linewidth=2, color = color)
    plt.fill_between(
        x, rmeans - rstds, rmeans + rstds, edgecolor='none', alpha=0.4, color = color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(fig_path)
    plt.close()

    if save_reward_path:
        with open(save_reward_path, 'w') as f:
            rmeans_str = ','.join([str(x) for x in rmeans])
            std_str = ','.join([str(x) for x in rstds])
            f.write(rmeans_str + '\n')
            f.write(std_str + '\n')

def rescale(x = None, min_val = None, max_val = None):

    scale_factor = max_val - min_val

    rescaled_x = float(x - min_val)/float(max_val-min_val)

    print(x, scale_factor, rescaled_x)

    return rescaled_x

def read_rewards_file(fname = None):
    with open(fname, 'r') as f:
        x = f.readlines()

        print(x)

        rmeans = [float(y) for y in x[0].strip().split(',')]
        rstds = [float(y) for y in x[1].strip().split(',')]

    min_reward = min(rmeans)
    max_reward = max(rmeans)

    min_std = min(rstds)
    max_std = max(rstds)

    scaled_reward = [rescale(x = r, min_val = min_reward, max_val = max_reward) for r in rmeans]

    min_scaled_reward = min(scaled_reward)
    max_scaled_reward = max(scaled_reward)

    scaled_std = []

    for i, unscaled_reward_val in enumerate(rmeans):

        try:
            fraction_std = float(rstds[i])/float(max_reward - min_reward)
        except:
            fraction_std = 0.0

        scaled_std.append(fraction_std * float(max_scaled_reward - min_scaled_reward))

    # both do not work
    # scaled_std = [rescale(x = r, min_val = min_std, max_val = max_std) for r in rstds]

    return np.array(rmeans), np.array(rstds), np.array(scaled_reward), np.array(scaled_std)

if __name__ == "__main__":

    args = parse_args()

    KL_results_dir = args.KL_results_dir

    entropy_results_dir = args.entropy_results_dir

    output_results_dir = args.output_results_dir

    config_file = args.config_file

    xmax = int(args.xmax)

    # get rewards per type
    ######################################
    env_name = 'follower'
    KL_rewards_file = KL_results_dir + '/values' +  str(env_name) + '_reward.txt'
    entropy_rewards_file = entropy_results_dir + '/values' +  str(env_name) + '_reward.txt'

    print('KL')
    KL_unscaled_reward, KL_unscaled_std, KL_scaled_reward, KL_scaled_std = read_rewards_file(fname = KL_rewards_file)
    print(' ')
    print(' ')

    print('entropy')
    entropy_unscaled_reward, entropy_unscaled_std, entropy_scaled_reward, entropy_scaled_std = read_rewards_file(fname = entropy_rewards_file)
    print(' ')
    print(' ')

    # plot scaled rewards
    ######################################
    scaled_KL_reward_plot  = output_results_dir + '/scaled_KL_reward.pdf'
    scaled_entropy_reward_plot  = output_results_dir + '/scaled_entropy_reward.pdf'
    plot_train_rewards(rmeans = KL_scaled_reward, rstds = KL_scaled_std, fig_path=scaled_KL_reward_plot)
    plot_train_rewards(rmeans = entropy_scaled_reward, rstds = entropy_scaled_std, fig_path=scaled_entropy_reward_plot, color = 'green')

    # plot unscaled rewards
    ######################################
    unscaled_KL_reward_plot  = output_results_dir + '/unscaled_KL_reward.pdf'
    unscaled_entropy_reward_plot  = output_results_dir + '/unscaled_entropy_reward.pdf'

    plot_train_rewards(rmeans = KL_unscaled_reward, rstds = KL_unscaled_std, fig_path = unscaled_KL_reward_plot)
    plot_train_rewards(rmeans = entropy_unscaled_reward, rstds = entropy_unscaled_std, fig_path = unscaled_entropy_reward_plot, color = 'green')

    # plot overlay
    ######################################
    scaled_overlay_plot = output_results_dir + '/scaled_overlaid_KL_entropy_reward.pdf'

    ymin = 0

    #title_str = r'true = $z_{1}$, $\alpha = 0.15$, $\beta = 1.0$'
    title_str = r'$\mathrm{true~} = \mathtt{surveil}_{z_{1}}$, $\beta_{C} = 0.15$, $\beta_{I} = 1.0$'
    plot_overlaid_rewards(KL_rmeans = KL_scaled_reward,
                       KL_rstds = KL_scaled_std,
                       entropy_rmeans = entropy_scaled_reward,
                       entropy_rstds = entropy_scaled_std,
                       fig_path=scaled_overlay_plot,
                       ylim = [0,1.2],
                       title_str = title_str)
