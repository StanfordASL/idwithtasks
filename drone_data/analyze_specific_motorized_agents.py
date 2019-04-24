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
import re
import pickle
import itertools
from collections import OrderedDict

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

FONT_SIZE = 20
LEGEND_FONT_SIZE = 20

XTICK_LABEL_SIZE = 24
YTICK_LABEL_SIZE = 24

AGENT_LW = 3.5
GLOBAL_LW = 5.5

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': XTICK_LABEL_SIZE,
         'ytick.labelsize': YTICK_LABEL_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)


from plotting_utils import *
from RL_utils import *
from parse_drone_data import *
from save_drone_data_pkl import *
from utils import *
from plot_campus_scene_from_pkl import *
from single_car_plot import *
from estimate_traffic_density import *

def parse_args():
    plot_results_dir = '~/idwithtasks/RL/drone_data/plots/subset_single_agent_scenes/'
    pkl_results_dir = '~/idwithtasks/RL/drone_data/plots/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_results_dir', type=str, required=False, default = plot_results_dir,  help="")
    parser.add_argument('--pkl_results_dir', type=str, required=False, default = pkl_results_dir,  help="")
    parser.add_argument('--conf_file', type=str, required=True, default = pkl_results_dir,  help="")

    args = parser.parse_args()
    return args

"""
    given pkl objects of all frames, trajectories per member, plot campus scenes and speeds of agents
"""

def plot_scene_from_pkl(pkl_results_dir = None, motorized_agent_id = None, scene_name = None, video_num = None, check_quality = None, min_distance_bound = None, plot_results_dir = None, pub_quality_mode = True, desired_other_agents = None):

    # load the pkl for agent of interest
    motorized_agent_pkl_path = pkl_results_dir + '/agent_' + str(motorized_agent_id) + '_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
    member_traj = load_pkl(pkl_path = motorized_agent_pkl_path)

    # plotting subfunction
    #################################################
    # plot the motorized agent
    example_traj = member_traj[motorized_agent_id][0]
    agent_type = example_traj[0]['class']
    member_id = example_traj[0]['member_id']

    motorized_time_bounds = [example_traj[0]['frame'], example_traj[-1]['frame']]

    # motorized agent trajectory
    distance_covered, full_x_vec, full_y_vec, min_distance_pass, agent_traj = get_agent_traj(example_traj = example_traj, check_quality = check_quality, min_distance_bound = min_distance_bound, frame_bounds = None)

    biker_traj_list = member_traj[motorized_agent_id][1:]

    # plot all the bikers
    for biker_traj in biker_traj_list:

        biker_type = biker_traj[0]['class']

        biker_member_id = biker_traj[0]['member_id']

        biker_time_bounds = [biker_traj[0]['frame'], biker_traj[-1]['frame']]

        biker_distance_covered, biker_x_vec, biker_y_vec, biker_min_distance_pass, biker_traj = get_agent_traj(example_traj = biker_traj, check_quality = check_quality, min_distance_bound = min_distance_bound, frame_bounds = motorized_time_bounds)

        if not pub_quality_mode:
            local_label = '_'.join([biker_type, str(biker_member_id)])
            plt.plot(biker_x_vec, biker_y_vec, label = local_label, ls = '--')
        else:
            plt.plot(biker_x_vec, biker_y_vec, ls = '--')
        plt.hold(True)

    # replot last one
    if not pub_quality_mode:
        local_label = '_'.join([biker_type, str(biker_member_id)])
        plt.plot(biker_x_vec, biker_y_vec, label = local_label, ls = '--', lw = AGENT_LW)
    else:
        plt.plot(biker_x_vec, biker_y_vec, ls = '--', label = ', '.join(desired_other_agents), lw = AGENT_LW)

    # plot our agent
    if not pub_quality_mode:
        local_label = '_'.join([agent_type, str(member_id)])
        plt.plot(full_x_vec, full_y_vec, label = local_label, lw = GLOBAL_LW, color='black')
    else:
        plt.plot(full_x_vec, full_y_vec, lw = GLOBAL_LW, color='black', label = agent_type)
    plt.hold(True)

    agent_str = '_'.join([agent_type] + desired_other_agents)
    frame_str = '_'.join([str(x) for x in motorized_time_bounds])

    if not pub_quality_mode:
        title_str = ' '.join(['frame: ', frame_str, agent_str])
        plt.title(title_str)
        plt.legend()
    else:
        plt.legend([', '.join(desired_other_agents), agent_type])

    plot_file = plot_results_dir + '/' + '_'.join(['MotorizedScene', 'frame', frame_str, 'member', str(member_id), 'scene', str(scene_name), 'video', str(video_num), agent_str]) + '.pdf'
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    #################################################
    return member_traj, example_traj, agent_type, full_x_vec, full_y_vec

def plot_speed_distance_panel(motorized_traj = None, motorized_agent_id = None, check_quality = None, scene_name = None, video_num = None, plot_results_dir = None, scene = None, desired_other_agents = ['Biker', 'Pedestrian'], min_distance_plot = False, pub_quality_mode = True, full_x_vec = None, full_y_vec = None, speed_xlim = None):

    # plot speed vs time
    speed_vec = get_agent_velocity_vector(member_traj_list = motorized_traj, check_quality = check_quality)

    # plot speed ts alone
    if pub_quality_mode:
        title_str = ''
    else:
        title_str = '_'.join([agent_type, str(motorized_agent_id)])

    speed_ylim = [0,15]
    speed_plot_file = plot_results_dir + '/plotSpeed_' + scene_name + '_agent_' + motorized_agent_id + '_video_' + str(video_num) + title_str + '.pdf'

    basic_plot_ts(ts_vector = speed_vec, title_str = title_str, plot_file = speed_plot_file, ylabel = r'speed $v$', ylim = speed_ylim)

    # plot speed, x coord, y coord
    speed_xy_dict = {}
    speed_xy_dict[r"speed $v$"] = {'ts_vector': speed_vec, 'ylim': speed_ylim, 'xlim': speed_xlim}
    speed_xy_dict[r'$x$'] = {'ts_vector': full_x_vec, 'ylim': None, 'xlim': speed_xlim}
    speed_xy_dict[r'$y$'] = {'ts_vector': full_y_vec, 'ylim': None, 'xlim': speed_xlim}

    speed_xy_plot_file = plot_results_dir + '/speedxy_' + scene_name + '_agent_' + motorized_agent_id + '_video_' + str(video_num) + title_str + '.pdf'
    plot_grid(normalized_ts_dict = speed_xy_dict, title_str = title_str, plot_file = speed_xy_plot_file)

    # plot speed vs other agents
    speed_vs_distance_dict = {}

    if min_distance_plot:
        speed_vec, min_distance_vec = plot_speed_vs_distance(example_traj = motorized_traj, scene_dict = scene, scene_name = scene_name, desired_other_agents = desired_other_agents, check_quality = check_quality)

        min_distance_ylim = [0, 450]
        speed_vs_distance_dict[r'speed $v$'] = {'ts_vector': speed_vec, 'ylim': speed_ylim, 'xlim': None}
        speed_vs_distance_dict[r'$d_{min}$'] = {'ts_vector': min_distance_vec, 'ylim': min_distance_ylim, 'xlim': None}

        speed_distance_plot_file = plot_results_dir + '/speed_vs_distance_' + scene_name + '_video_' + str(video_num) + title_str + '.pdf'
        plot_grid(normalized_ts_dict = speed_vs_distance_dict, title_str = title_str, plot_file = speed_distance_plot_file)

    return speed_xy_dict, speed_vs_distance_dict

if __name__ == "__main__":

    args = parse_args()

    pkl_results_dir = args.pkl_results_dir
    all_plot_results_dir = args.plot_results_dir
    conf_file = args.conf_file

    # file path configs
    ######################################
    file_path_config = ConfigParser.ConfigParser()
    file_path_config.read(conf_file)
    print file_path_config.sections()

    # get the scenes and agents of interest
    # scene_video_list = ['deathCircle_0', 'deathCircle_1', 'deathCircle_3']
    scene_video_list = file_path_config.get('SCENE_PARAMS', 'scene_list').split(',')

    # agents_of_interest = ['193', '442', '429']
    agents_of_interest = file_path_config.get('SCENE_PARAMS', 'agents_of_interest').split(',')

    # agents = ['Cart', 'Car', 'Bus']
    agents = file_path_config.get('SCENE_PARAMS', 'agents').split(',')

    # other_agents = ['Biker', 'Pedestrian']
    other_agents = file_path_config.get('SCENE_PARAMS', 'other_agents').split(',')

    # xmax
    speed_xmax = file_path_config.get('SCENE_PARAMS', 'speed_xmax')

    if speed_xmax != 'None':
        speed_xlim = [0,float(speed_xmax)]
    else:
        speed_xlim = None

    for scene_video in scene_video_list:

        scene_name = scene_video.split('_')[0]
        video_num = int(scene_video.split('_')[1])

        plot_results_dir = all_plot_results_dir + '/' + scene_name + '_' + str(video_num)
        remove_and_create_dir(plot_results_dir)

        # member traj has to be at least this high
        MIN_DISTANCE_COVERED = 700

        CLOSE_AGENT_DISTANCE = 200

        check_quality_global = False

        plot_axes = None

        pub_quality_mode = True
        check_quality_speed_plot = True
        min_distance_plot = True

        for motorized_agent_id in agents_of_interest:
            all_member_traj, motorized_traj, agent_type, full_x_vec, full_y_vec = plot_scene_from_pkl(pkl_results_dir = pkl_results_dir, motorized_agent_id = motorized_agent_id, scene_name = scene_name, video_num = video_num, check_quality = check_quality_global, min_distance_bound = MIN_DISTANCE_COVERED, plot_results_dir = plot_results_dir, desired_other_agents = ['Biker', 'Pedestrian'], pub_quality_mode = pub_quality_mode)

            # plot a series of speeds vs trajectories etc.

            if min_distance_plot:
                scene_pkl_path = pkl_results_dir + '/frameInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
                scene = load_pkl(pkl_path = scene_pkl_path)
            else:
                scene = None

            speed_xy_dict, speed_vs_distance_dict = plot_speed_distance_panel(motorized_traj = motorized_traj, motorized_agent_id = motorized_agent_id, check_quality = check_quality_speed_plot, scene_name = scene_name, video_num = video_num, plot_results_dir = plot_results_dir, scene = scene, desired_other_agents = ['Biker', 'Pedestrian'], min_distance_plot = min_distance_plot, pub_quality_mode = pub_quality_mode, full_x_vec = full_x_vec, full_y_vec = full_y_vec, speed_xlim = speed_xlim)

            # now get the local traffic density
            # frame_rho, density_ts = traffic_density_near_agent(motorized_traj = motorized_traj, desired_agents = ['Biker', 'Pedestrian'], scene = scene, min_distance = CLOSE_AGENT_DISTANCE)
            frame_rho = all_traffic_statistics_near_agent(motorized_traj = motorized_traj, desired_agents = ['Biker', 'Pedestrian'], all_member_traj = all_member_traj, min_distance = CLOSE_AGENT_DISTANCE, quality_check = True, construct_from_scene_pkl = True, scene = scene)

            # now plot density ts, min distances vs agent speed
            speed_ts = [x['speed'] for x in frame_rho.values()]
            x_ts = [x['x'] for x in frame_rho.values()]
            y_ts = [x['y'] for x in frame_rho.values()]
            min_distance_ts = [x['min_distance'] for x in frame_rho.values()]
            max_distance_ts = [x['max_distance'] for x in frame_rho.values()]
            rho_ts = [x['density_rho'] for x in frame_rho.values()]
            circle_rho_ts = [x['all_circle_density'] for x in frame_rho.values()]

            frame_seq = frame_rho.keys()

            speed_ylim = [0,12]
            d_ylim = [0,150]
            rho_ylim = [0,12]
            circle_rho_ylim = [0,30]

            rho_d_dict = {}
            rho_d_dict[r'speed $v$'] = {'ts_vector': speed_ts, 'ylim': speed_ylim, 'xlim': None, 'x': frame_seq}
            rho_d_dict[r'$x$'] = {'ts_vector': x_ts, 'ylim': None, 'xlim': None, 'x': frame_seq}
            rho_d_dict[r'$y$'] = {'ts_vector': y_ts, 'ylim': None, 'xlim': None, 'x': frame_seq}
            #rho_d_dict[r'$d_{min}$'] = {'ts_vector': min_distance_ts, 'ylim': None, 'xlim': None, 'x': frame_seq}
            #rho_d_dict[r'$d_{max}$'] = {'ts_vector': max_distance_ts, 'ylim': None, 'xlim': None}
            #rho_d_dict[r'$\rho$'] = {'ts_vector': rho_ts, 'ylim': None, 'xlim': None, 'x':frame_seq}

            title_str = ' , '.join([scene_name + str(video_num), 'agent: ' + str(motorized_agent_id)])
            plot_file = plot_results_dir + '/FrameSpeed_distance_' + scene_name + '_video_' + str(video_num) + '_agent_' + str(motorized_agent_id) + '.pdf'
            plot_grid(normalized_ts_dict = rho_d_dict, title_str = title_str, plot_file = plot_file)


            rho_d_dict = {}
            rho_d_dict[r'speed $v$'] = {'ts_vector': speed_ts, 'ylim': speed_ylim, 'xlim': None, 'x': frame_seq}
            rho_d_dict[r'$d_{min}$'] = {'ts_vector': min_distance_ts, 'ylim': d_ylim, 'xlim': None, 'x': frame_seq}
            rho_d_dict[r'$\rho$'] = {'ts_vector': rho_ts, 'ylim': rho_ylim, 'xlim': None, 'x':frame_seq}
            rho_d_dict[r'circle $\rho$'] = {'ts_vector': circle_rho_ts, 'ylim': circle_rho_ylim, 'xlim': None, 'x':frame_seq}

            plot_file = plot_results_dir + '/FrameRho_distance_' + scene_name + '_video_' + str(video_num) + '_agent_' + str(motorized_agent_id) + '.pdf'
            plot_grid(normalized_ts_dict = rho_d_dict, title_str = title_str, plot_file = plot_file)
