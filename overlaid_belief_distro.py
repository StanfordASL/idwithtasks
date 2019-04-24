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

CDC_ROOT_DIR = os.environ['CDC_ROOT_DIR']
sys.path.append(CDC_ROOT_DIR)
sys.path.append(CDC_ROOT_DIR + '/RL/')

from plotting_utils import *
from RL_utils import *

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 34}
plt.rc('font', **font)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--KL_results_dir', type=str, required=True, help="")

    parser.add_argument('--H_results_dir', type=str, required=True, help="")

    parser.add_argument('--output_results_dir', type=str, default=None, required=False, help="")

    parser.add_argument('--config_file', type=str, required=True, help="")

    parser.add_argument('--xmax', type=int, required=False, default = 20, help="")

    args = parser.parse_args()
    return args

def parse_RL_results(results_csv = None):

    #####################################
    results_df = pandas.read_csv(results_csv)

    unique_test_episodes = list(set(results_df['episode_number']))

    # get diff batch numbers
    unconverged_episode = min(unique_test_episodes)

    converged_episode = max(unique_test_episodes)

    # get different batches
    unconverged_df = results_df[results_df['episode_number'] == unconverged_episode]

    converged_df = results_df[results_df['episode_number'] == converged_episode]

    return converged_df


if __name__ == "__main__":

    args = parse_args()

    KL_results_dir = args.KL_results_dir
    H_results_dir = args.H_results_dir

    config_file = args.config_file
    output_results_dir = args.output_results_dir
    xmax = int(args.xmax)

    if not output_results_dir:
        output_results_dir = base_results_dir

    # file path configs
    ######################################
    file_path_config = ConfigParser.ConfigParser()
    file_path_config.read(config_file)
    print file_path_config.sections()

    # reward params
    #####################################
    reward_params_dict = {}
    control_weight = file_path_config.getfloat('REWARD_PARAMS', 'control_weight')
    entropy_weight = file_path_config.getfloat('REWARD_PARAMS', 'entropy_weight')

    reward_params_dict['control_weight'] = control_weight
    reward_params_dict['entropy_weight'] = entropy_weight

    model_names = file_path_config.get('MODEL_PARAMS', 'model_names').split(',')
    true_model = file_path_config.get('MODEL_PARAMS', 'true_model')

    print('model_names', model_names)


    KL_results_csv  = KL_results_dir + '/results.csv'
    KL_df = parse_RL_results(results_csv = KL_results_csv)

    H_results_csv  = H_results_dir + '/results.csv'
    H_df = parse_RL_results(results_csv = H_results_csv)

    # converged df
    xlim = [0, xmax]
    plot_paired_episode(KL_df = KL_df, H_df = H_df, true_model = true_model, batch_descriptor = 'converged', base_results_dir = output_results_dir, reward_params_dict = reward_params_dict, model_names = model_names, xlim = xlim)
