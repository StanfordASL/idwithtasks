import numpy as np
from os import path
import cPickle
import sys, os
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from plotting_utils import *

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

FONT_SIZE = 24
LEGEND_FONT_SIZE = 15

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': FONT_SIZE,
         'ytick.labelsize': FONT_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)








# episode
# iteration
# belief_vector
# action_taken
# control cost, entropy diff

def report_rewards(
                    reward = None,
                    episode_number=None,
                    iteration_index=None,
                    belief_str=None,
                    model_str=None,
                    control_action=None,
                    results_df = None,
                    weighted_control_cost = None,
                    weighted_info_gain = None,
                    info_gain = None,
                    control_cost = None,
                    control_weight = None,
                    entropy_weight = None,
                    entropy_KL = None,
                    true_model = None):

    # dictionary of useful data
    df_dict = {}
    df_dict['reward'] = [reward]
    df_dict['episode_number'] = [episode_number]
    df_dict['iteration_index'] = [iteration_index]
    df_dict['belief_str'] =  [belief_str]
    df_dict['model_str'] =  [model_str]
    df_dict['control_action'] =  [control_action]

    # new ones
    df_dict['weighted_control_cost'] =  [weighted_control_cost]
    df_dict['weighted_info_gain'] =  [weighted_info_gain]
    df_dict['control_cost'] =  [control_cost]
    df_dict['info_gain'] =  [info_gain]
    df_dict['entropy_weight'] =  [entropy_weight]
    df_dict['control_weight'] =  [control_weight]
    df_dict['entropy_KL'] =  [entropy_KL]
    df_dict['true_model'] =  [true_model]

    # convert to df
    local_df = pandas.DataFrame(df_dict)
    # append to existing dataframe of previous time results
    results_df = results_df.append(local_df)
    return results_df

def belief_str_to_vec(belief_str):

    belief_str_vec = belief_str.split('_')

    belief_vec = np.array([float(x) for x in belief_str_vec])

    return belief_vec


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_results_dir', type=str, required=True, help="")
    args = parser.parse_args()
    return args


def plot_episode(df = None, true_model = None, batch_descriptor = 'converged', base_results_dir = None, reward_params_dict = None, model_names = None, xlim = None, ylim = None):

    optimal_action_list = list(df['control_action'])

    reward_list = list(df['reward'])

    belief_str_list = list(df['belief_str'])

    # B histories
    B_hist = [belief_str_to_vec(x) for x in belief_str_list]


    ########################
    plot_file = base_results_dir + '/belief_' + batch_descriptor +  '.pdf'


    # plot belief
    plot_belief_history(B_hist = B_hist, reward_params_dict = reward_params_dict, true_model = true_model, plot_file = plot_file, model_names = model_names, xlim = xlim, ylim = [0,1.0])

    # optimal action
    plot_file = base_results_dir + '/action_' + batch_descriptor +  '.pdf'
    plot_ts(ts_vector = optimal_action_list, reward_params_dict = reward_params_dict, true_model = true_model, plot_file = plot_file, ylabel = 'action', xlim = xlim)

    # plot reward
    plot_file = base_results_dir + '/reward_' + batch_descriptor +  '.pdf'
    plot_ts(ts_vector = reward_list, reward_params_dict = reward_params_dict, true_model = true_model, plot_file = plot_file, ylabel = 'reward', xlim = xlim)

    return B_hist, optimal_action_list, reward_list


def plot_paired_episode(KL_df = None, H_df = None, true_model = None, batch_descriptor = 'converged', base_results_dir = None, reward_params_dict = None, model_names = None, xlim = None, ylim = None):

    # KL B histories
    KL_belief_str_list = list(KL_df['belief_str'])
    KL_B_hist = [belief_str_to_vec(x) for x in KL_belief_str_list]

    # H B histories
    H_belief_str_list = list(H_df['belief_str'])
    H_B_hist = [belief_str_to_vec(x) for x in H_belief_str_list]

    ########################
    plot_file = base_results_dir + '/overlaid_KL_H_belief_' + batch_descriptor +  '.pdf'

    # plot belief
    plot_overlaid_belief_history(KL_B_hist = KL_B_hist, H_B_hist = H_B_hist, reward_params_dict = reward_params_dict, true_model = true_model, plot_file = plot_file, model_names = model_names, xlim = xlim, ylim = [0,1.0])

    return KL_B_hist, H_B_hist

def plot_overlaid_belief_history(KL_B_hist = None, H_B_hist = None, reward_params_dict = None, true_model = None, plot_file = None, model_names = None, ylabel = r'Belief $B_k$', linewidth=3.0, xlim = None, ylim = None):

    KL_B_hist = np.asarray(KL_B_hist)

    H_B_hist = np.asarray(H_B_hist)

    num_models = len(model_names)
    print(num_models)
    print(model_names)

    color_vec = ['blue', 'red', 'green', 'purple', 'yellow', 'cyan']

    model_names_latex = []
    for i, model_name in enumerate(model_names):

        latex_model_name = k_bound_latex_name(model_name = model_name, camera_ready=True)

        #plt.plot(KL_B_hist.T[i], linewidth=linewidth, color=color_vec[i], label = latex_model_name)
        plt.plot(KL_B_hist.T[i], linewidth=linewidth, color=color_vec[i], label = latex_model_name)
        plt.plot(H_B_hist.T[i], linewidth=linewidth, ls = '--', color=color_vec[i])
        plt.hold(True)

        model_names_latex.append(latex_model_name)

    #title_str = ' '.join(['true: ', true_model, 'control_wt: ', str(reward_params_dict['control_weight']), 'entropy_wt: ', str(reward_params_dict['entropy_weight'])])
    model_str = r"true = " + k_bound_latex_name(model_name = true_model, title =True)
    #alpha_str = r"$\alpha = " + str(reward_params_dict['control_weight']) + r"$"
    #beta_str = r"$\beta = " + str(reward_params_dict['entropy_weight']) + r"$"
    alpha_str = r"$\beta_{C} = " + str(reward_params_dict['control_weight']) + r"$"
    beta_str = r"$\beta_{I} = " + str(reward_params_dict['entropy_weight']) + r"$"
    title_str = ' , '.join([model_str, alpha_str, beta_str])

    plt.xlabel(r'Iteration $k$')

    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    #plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=2)
    plt.legend(bbox_to_anchor=(1.02, 0.65), ncol=2)

    plt.title(title_str)
    plt.savefig(plot_file)

    plt.close()



if __name__ == "__main__":

    args = parse_args()

    base_results_dir = args.base_results_dir

    results_csv = args.base_results_dir + '/results.csv'

    results_df = pandas.read_csv(results_csv)

    unique_test_episodes = list(set(results_df['episode_number']))

    # get diff batch numbers
    unconverged_episode = min(unique_test_episodes)

    converged_episode = max(unique_test_episodes)

    # get different batches
    unconverged_df = results_df[results_df['episode_number'] == unconverged_episode]

    converged_df = results_df[results_df['episode_number'] == converged_episode]

    # reward params
    reward_params_dict = {}
    reward_params_dict['control_weight'] = 0.10
    reward_params_dict['entropy_weight'] = 1.0

    model_names = ['benign',
                    'k-bound',
                    'pursuer']

    true_model = 'k-bound'

    # converged df
    plot_episode(df = converged_df, true_model = true_model, batch_descriptor = 'converged', base_results_dir = base_results_dir, reward_params_dict = reward_params_dict, model_names = model_names)

    # unconverged
    #plot_episode(df = unconverged_df, true_model = true_model, batch_descriptor = 'unconverged', base_results_dir = base_results_dir, reward_params_dict = reward_params_dict, model_names = model_names)
