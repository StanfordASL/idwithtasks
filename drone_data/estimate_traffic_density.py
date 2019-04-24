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

CDC_ROOT_DIR = os.environ['CDC_ROOT_DIR']
sys.path.append(CDC_ROOT_DIR)
sys.path.append(CDC_ROOT_DIR + '/RL/')

from plotting_utils import *
from RL_utils import *
from parse_drone_data import *
from save_drone_data_pkl import *
from utils import *
from plot_campus_scene_from_pkl import *

def construct_close_agents_dict_from_scene_pkl(scene = None, frame = None, agent_xy_tuple = None, desired_agents = None, quality_met = None, min_distance = None, region_bounds = [[400, 1000], [750, 1250]]):

    x_bound = region_bounds[0]
    y_bound = region_bounds[1]

    if frame in scene.keys():
        # now look at all other agents

        scene_info = scene[frame]

        close_agents_dict = {}

        in_region_list = []
        # get num unique agents per agent type
        for other_agent_dict in scene_info:
            other_agent_type = other_agent_dict['class']
            other_member_id = other_agent_dict['member_id']
            other_agent_xy_tuple = other_agent_dict['xy']

            # is an agent one we care about and in box of interest?
            desired_agent_bool = (other_agent_type in desired_agents)

            agent_distance = calc_distance(xy_one = agent_xy_tuple, xy_two = other_agent_xy_tuple)

            in_distance_bool = (agent_distance <= min_distance)

            if desired_agent_bool and in_distance_bool and quality_met:
                close_agents_dict[other_member_id] = agent_distance

            # see if other agent in area of interest
            # is an agent one we care about and in box of interest?
            in_x_bool = (other_agent_xy_tuple[0] >= x_bound[0]) and (other_agent_xy_tuple[0] <= x_bound[1])
            in_y_bool = (other_agent_xy_tuple[1] >= y_bound[0]) and (other_agent_xy_tuple[1] <= y_bound[1])
            in_region_bool = in_x_bool and in_y_bool

            if desired_agent_bool and in_region_bool and quality_met:
                in_region_list.append(other_member_id)

        total_density = len(set(close_agents_dict.keys()))
        all_circle_density = len(set(in_region_list))

    else:
        close_agents_dict = {}
        total_density = np.nan
        all_circle_density = np.nan

    return close_agents_dict, total_density, all_circle_density


def get_instantaneous_velocity(motorized_traj = None, i = None, quality_check = True):
    quality_met = False

    if i < len(motorized_traj) - 1:

        current_coordinate = motorized_traj[i]['xy']
        next_coordinate = motorized_traj[i+1]['xy']

        current_frame = int(motorized_traj[i]['frame'])
        next_frame = int(motorized_traj[i+1]['frame'])

        if(next_frame == (current_frame + 1) ):
            xdot = next_coordinate[0] - current_coordinate[0]
            ydot = next_coordinate[1] - current_coordinate[1]

            agent_speed = ((xdot)**2 + (ydot)**2)**(0.5)
            agent_velocity_vec = (xdot, ydot)

        quality_met = (motorized_traj[i]['quality'] == '0_0') and (motorized_traj[i+1]['quality'] == '0_0')

        if(quality_check):
            if(quality_met):
                pass
            else:
                agent_speed = np.nan
                agent_velocity_vec = (np.nan, np.nan)
    else:
        agent_speed = np.nan
        agent_velocity_vec = (np.nan, np.nan)

    return agent_speed, agent_velocity_vec


def get_velocity_vectors_all_agents(all_member_traj = None, quality_check = False):

    velocity_dict = {}
    frame_dict = {}
    agent_len_dict = {}

    # key is member_id
    # value is single_agent_traj_list
    # single_agent_traj_list = [{'frame', 'xy', speed, 'xdotydot'}, ...]

    # to plot a single agent, index by key and loop through frames

    # to plot a frame, loop through all agents, get frame of interest only

    all_agent_traj_list = all_member_traj[all_member_traj.keys()[0]]

    for single_agent_traj_list in all_agent_traj_list:

        member_id = single_agent_traj_list[0]['member_id']

        velocity_dict[member_id] = []

        agent_len_dict[member_id] = len(single_agent_traj_list)

        for frame_index, single_agent_traj in enumerate(single_agent_traj_list):

            frame = single_agent_traj['frame']
            xy = single_agent_traj['xy']
            member_id = single_agent_traj['member_id']
            quality = single_agent_traj['quality']
            agent_class = single_agent_traj['class']

            # get the speed and velocity vector at the current frame index
            speed, xy_dot = get_instantaneous_velocity(motorized_traj = single_agent_traj_list, i = frame_index, quality_check = quality_check)

            if quality_check:
                if quality == '0_0':
                    velocity_dict[member_id].append({'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality})
            else:
                velocity_dict[member_id].append({'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality})

            # curr frame list
            #####################################
            if frame in frame_dict.keys():
                curr_frame_list = frame_dict[frame]

                if quality_check:
                    if quality == '0_0':
                        curr_frame_list.append({'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality, 'class': agent_class})

                else:
                    curr_frame_list.append({'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality, 'class': agent_class})

            else:

                if quality_check:
                    if quality == '0_0':
                        curr_frame_list = [{'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality, 'class': agent_class}]
                    else:
                        curr_frame_list = []

                else:
                        curr_frame_list = [{'frame': frame, 'xy': xy, 'speed': speed, 'xy_dot': xy_dot, 'id': member_id, 'quality': quality, 'class': agent_class}]

            frame_dict[frame] = curr_frame_list

    return velocity_dict, frame_dict, agent_len_dict


# as the agent moves along, how many other agents are with a min distance of it?
def all_traffic_statistics_near_agent(motorized_traj = None, desired_agents = ['Biker', 'Pedestrian'], all_member_traj = None, min_distance = None, quality_check = False, construct_from_scene_pkl = True, scene = None):

    # how motorized traj looks
    # {'lost': '0', 'occluded': '0', 'frame': 6500, 'generated': '0', 'xy': (1075.0, 838.5), 'member_id': '39', 'quality': '0_0', 'class': 'Car'}

    # how all member traj looks
    # >>> all_member_traj['39'][-1][0]
    # {'lost': '1', 'occluded': '0', 'frame': 6500, 'generated': '0', 'xy': (793.0, 31.5), 'member_id': '5', 'quality': '1_0', 'class': 'Biker'}

    frame_rho = {}

    # keys: frame number
    # value: min_dist, rho, speed, x, y

    for i, motorized_frame in enumerate(motorized_traj):

        frame = motorized_frame['frame']

        quality_met = True

        if (quality_check):
            quality_met = (motorized_frame['quality'] == '0_0')

        agent_xy_tuple = motorized_frame['xy']

        if construct_from_scene_pkl:
            close_agents_dict, total_density, all_circle_density = construct_close_agents_dict_from_scene_pkl(scene = scene, frame = frame, agent_xy_tuple = agent_xy_tuple, desired_agents = desired_agents, quality_met = quality_met, min_distance = min_distance)

        else:
            print('not supported, construct from other member trajectories')
            close_agents_dict = {}
            total_density = np.nan

        # get the speed
        frame_info_dict = {}

        agent_speed, agent_velocity_vec = get_instantaneous_velocity(motorized_traj = motorized_traj, i = i, quality_check = quality_check)

        frame_info_dict['close_agents_dict'] = close_agents_dict
        frame_info_dict['density_rho'] = total_density
        frame_info_dict['x'] = agent_xy_tuple[0]
        frame_info_dict['y'] = agent_xy_tuple[1]
        frame_info_dict['speed'] = agent_speed
        frame_info_dict['xy'] = agent_xy_tuple
        frame_info_dict['xy_dot'] = agent_velocity_vec
        frame_info_dict['all_circle_density'] = all_circle_density

        if close_agents_dict != {}:
            distance_vec = close_agents_dict.values()

            frame_info_dict['min_distance'] = np.min(distance_vec)
            frame_info_dict['max_distance'] = np.max(distance_vec)
        else:
            frame_info_dict['min_distance'] = np.nan
            frame_info_dict['max_distance'] = np.nan

        frame_rho[frame] = frame_info_dict



    return frame_rho
    # plot rho near agent vs agent speed


# as the agent moves along, how many other agents are with a min distance of it?
def traffic_density_near_agent(motorized_traj = None, desired_agents = ['Biker', 'Pedestrian'], scene = None, min_distance = None, quality_check = False, construct_from_scene_pkl = True):

    # how motorized traj looks
    # {'lost': '0', 'occluded': '0', 'frame': 6500, 'generated': '0', 'xy': (1075.0, 838.5), 'member_id': '39', 'quality': '0_0', 'class': 'Car'}

    # how all member traj looks
    # >>> all_member_traj['39'][-1][0]
    # {'lost': '1', 'occluded': '0', 'frame': 6500, 'generated': '0', 'xy': (793.0, 31.5), 'member_id': '5', 'quality': '1_0', 'class': 'Biker'}

    frame_rho = {}

    for motorized_frame in motorized_traj:

        frame = motorized_frame['frame']

        quality_met = True

        if (quality_check):
            quality_met = (motorized_frame['quality'] == '0_0')

        agent_xy_tuple = motorized_frame['xy']

        if construct_from_scene_pkl:
            close_agents_dict, total_density = construct_close_agents_dict_from_scene_pkl(scene = scene, frame = frame, agent_xy_tuple = agent_xy_tuple, desired_agents = desired_agents, quality_met = quality_met, min_distance = min_distance)

        else:
            print('not supported, construct from other member trajectories')



        frame_rho[frame] = {'close_agents_dict': close_agents_dict, 'density_rho': total_density}

    # plot rho vs frame
    ####################################
    density_ts = [x['density_rho'] for x in frame_rho.values()]

    return frame_rho, density_ts
    # plot rho near agent vs agent speed

def estimate_traffic_density(region_bounds = [[400, 1000], [750, 1500]], desired_agents = ['Biker', 'Pedestrian'], scene = None, frame_bounds = None, print_mode = False):

    x_bound = region_bounds[0]
    y_bound = region_bounds[1]

    # this is for the full dictionary and only populated for data in frame_bounds if it exists
    density_ts = []

    # now create a selective dict from [frame] to [rho] if rho can be calculated for that frame
    density_dict = {}

    for frame, scene_info in scene.iteritems():

        member_list = []

        in_bound = False
        try:
            in_bound = ( (frame >= frame_bounds[0]) and (frame <= frame_bounds[1]) )
        except:
            in_bound = True

        # check if we care about this frame
        if ( in_bound ):

            # get num unique agents per agent type
            for agent_dict in scene_info:
                agent_type = agent_dict['class']
                member_id = agent_dict['member_id']
                agent_x = agent_dict['xy'][0]
                agent_y = agent_dict['xy'][1]

                # is an agent one we care about and in box of interest?
                desired_agent_bool = (agent_type in desired_agents)
                in_x_bool = (agent_x >= x_bound[0]) and (agent_x <= x_bound[1])
                in_y_bool = (agent_y >= y_bound[0]) and (agent_y <= y_bound[1])
                in_region_bool = in_x_bool and in_y_bool

                if desired_agent_bool and in_region_bool:
                    member_list.append(member_id)

                    if print_mode:
                        print('type', agent_type)
                        print('member_id', member_id)
                        print('agent_x', agent_x)
                        print('agent_y', agent_y)

        num_unique_members = len(set(member_list))
        density_ts.append(num_unique_members)

        density_dict[frame] = num_unique_members

    return density_ts, density_dict

def parse_args():
    plot_results_dir = '~/idwithtasks/RL/drone_data/plots/traffic_densities/'
    pkl_results_dir = '~/idwithtasks/RL/drone_data/plots/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_results_dir', type=str, required=False, default = plot_results_dir,  help="")
    parser.add_argument('--pkl_results_dir', type=str, required=False, default = pkl_results_dir,  help="")
    parser.add_argument('--conf_file', type=str, required=True, default = pkl_results_dir,  help="")

    args = parser.parse_args()
    return args

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

    # ['gates_1', 'gates_6', 'gates_8']
    scene_video_list = file_path_config.get('SCENE_PARAMS', 'scene_list').split(',')

    region_x = [400, 1000]
    region_y = [750, 1500]
    region_bounds = [region_x, region_y]
    ylim = [0,40]

    for scene_video in scene_video_list:

        scene_name = scene_video.split('_')[0]
        video_num = int(scene_video.split('_')[1])

        plot_results_dir = all_plot_results_dir + '/' + scene_name + '_' + str(video_num)
        remove_and_create_dir(plot_results_dir)

        # pkl paths for saved dictionaries
        scene_pkl_path = pkl_results_dir + '/frameInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        frameList_pkl_path = pkl_results_dir + '/frameList_' + scene_name + '_video_' + str(video_num) + '.pkl'

        print('loading scene')
        scene = load_pkl(pkl_path = scene_pkl_path)
        print('loading frame lists')
        all_frame_bounds = load_pkl(pkl_path = frameList_pkl_path)

        desired_agents = ['Biker', 'Pedestrian']

        #######################################
        density_ts, density_dict = estimate_traffic_density(region_bounds = region_bounds, desired_agents = desired_agents, scene = scene, frame_bounds = None, print_mode = False)

        agent_str = '_'.join(desired_agents)
        frame_str = 'all frames'
        title_str = ' '.join(['frame: ', frame_str, agent_str])
        plt.title(title_str)

        plot_file = plot_results_dir + '/' + '_'.join(['TrafficDensity', 'frame', frame_str, 'scene', str(scene_name), 'video', str(video_num), agent_str]) + '.pdf'

        basic_plot_ts(ts_vector = np.array(density_ts), title_str = title_str, plot_file = plot_file, ylabel = 'Num_agents', ylim = ylim)


        # look at all [frame_start, frame_end] where we have a motorized agent
        for frame_bounds in all_frame_bounds:

            density_ts, local_density_dict = estimate_traffic_density(region_bounds = region_bounds, desired_agents = desired_agents, scene = scene, frame_bounds = frame_bounds, print_mode = False)

            agent_str = '_'.join(desired_agents)
            frame_str = '_'.join([str(x) for x in frame_bounds])
            title_str = ' '.join(['frame: ', frame_str, agent_str])
            plt.title(title_str)

            plot_file = plot_results_dir + '/' + '_'.join(['TrafficDensity', 'frame', frame_str, 'scene', str(scene_name), 'video', str(video_num), agent_str]) + '.pdf'

            basic_plot_ts(ts_vector = np.array(density_ts), title_str = title_str, plot_file = plot_file, ylabel = 'Num_agents', ylim = ylim)
