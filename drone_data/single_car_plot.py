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

from plotting_utils import *
from RL_utils import *
from parse_drone_data import *
from save_drone_data_pkl import *
from utils import *
from plot_campus_scene_from_pkl import *
from analyze_specific_motorized_agents import *

def parse_args():
    plot_results_dir = '~/idwithtasks/RL/drone_data/plots/single_agent_scenes/'
    pkl_results_dir = '~/idwithtasks/RL/drone_data/plots/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_results_dir', type=str, required=False, default = plot_results_dir,  help="")
    parser.add_argument('--pkl_results_dir', type=str, required=False, default = pkl_results_dir,  help="")
    
    args = parser.parse_args()
    return args

def in_frame_check(frame_bounds = None, traj_dict = None):
    
    in_frame_bool = (traj_dict['frame'] >= frame_bounds[0]) and (traj_dict['frame'] <= frame_bounds[1])
    return in_frame_bool
     
def get_xy_vec_from_traj(example_traj = None, check_quality = None):

    if check_quality:
        full_x_vec = [x['xy'][0] for x in example_traj if x['quality'] == '0_0']
        full_y_vec = [x['xy'][1] for x in example_traj if x['quality'] == '0_0']

    else:
        full_x_vec = [x['xy'][0] for x in example_traj]
        full_y_vec = [x['xy'][1] for x in example_traj]
    return full_x_vec, full_y_vec

def get_agent_traj(example_traj = None, check_quality = None, min_distance_bound = 0, frame_bounds = None):

    # limit trajectory to within frame
    if frame_bounds:
        frame_traj = [x for x in example_traj if in_frame_check(frame_bounds = frame_bounds, traj_dict = x)]
    else:
        frame_traj = example_traj

    full_x_vec, full_y_vec = get_xy_vec_from_traj(example_traj = frame_traj, check_quality = check_quality)

    xy_one = [full_x_vec[0], full_y_vec[0]]
    xy_two = [full_x_vec[-1], full_y_vec[-1]]

    distance_covered = calc_distance(xy_one = xy_one, xy_two = xy_two)
    min_distance_pass = float(distance_covered) >= float(min_distance_bound)

    return distance_covered, full_x_vec, full_y_vec, min_distance_pass, frame_traj

def get_time_overlap(time_bound_one = None, time_bound_two = None):

    ref_start = time_bound_one[0]
    ref_end = time_bound_one[1]

    comparison_start = time_bound_two[0]
    comparison_end = time_bound_two[1]

    assert(ref_end >= ref_start)
    assert(comparison_end >= comparison_start)

    if ( (comparison_end >= ref_start) and (comparison_start <= ref_end) ):
        time_overlap_boolean = True
    else:
        time_overlap_boolean = False

    return time_overlap_boolean


def plot_all_motorized_agent(traj_list = None, base_plot_dir = None, scene_name = None, video_num = None, desired_agents = None, other_agents = None, check_quality = None, plot_axes = None, min_distance_bound = None, pkl_results_dir = None):               

    for member_id, example_traj in traj_list.iteritems():
        agent_type = example_traj[0]['class']
        member_id = example_traj[0]['member_id']

        # for each motorized agent, check if it covers the minimum distance
        # - covers minimum distance?
        #  - does it pass the quality?
        #  - when does it start and end?
        # plot its trajectory from [START, END]
        # for each other agent:
            # - is it in other_agents?
            # - does it pass minimum distance
            # - plot its trajectory from [START, END]


        plot_specific_motorized_agent(specific_motorized_agent_id = member_id, traj_list = traj_list, base_plot_dir = base_plot_dir, scene_name = scene_name, video_num = video_num, desired_agents = desired_agents, other_agents = other_agents, check_quality = check_quality, plot_axes = plot_axes, min_distance_bound = min_distance_bound, pkl_results_dir = pkl_results_dir)              

"""
    specific motorized agent, need to know this a priori
"""

def plot_specific_motorized_agent(specific_motorized_agent_id = None, traj_list = None, base_plot_dir = None, scene_name = None, video_num = None, desired_agents = None, other_agents = None, check_quality = None, plot_axes = None, min_distance_bound = None, pkl_results_dir = None, plot_speed_mode = True, scene = None):  

    # list of agent of interest AND only bikers that intersect it 
    subset_traj_dict = {}
    subset_traj_list = []

    example_traj = traj_list[specific_motorized_agent_id]
    agent_type = example_traj[0]['class']
    member_id = example_traj[0]['member_id']

    # check bounds on motorized agent
    agent_type_pass = agent_type in desired_agents

    distance_covered, full_x_vec, full_y_vec, min_distance_pass, agent_traj = get_agent_traj(example_traj = example_traj, check_quality = check_quality, min_distance_bound = min_distance_bound, frame_bounds = None)

    subset_traj_list.append(agent_traj)

    motorized_time_bounds = [example_traj[0]['frame'], example_traj[-1]['frame']]

    # print('motorized bounds', motorized_time_bounds)

    agent_overall_pass = min_distance_pass and agent_type_pass

    # if motorized vehicle makes sense
    if (agent_overall_pass):
        
        per_agent_plot_dir = base_plot_dir + '/' + str(member_id)
        remove_and_create_dir(per_agent_plot_dir)

        # look at all other agents and plot their traj if in time bounds
        for biker_member_id, biker_traj in traj_list.iteritems():
            biker_type = biker_traj[0]['class']

            biker_time_bounds = [biker_traj[0]['frame'], biker_traj[-1]['frame']]
           
            # print('biker bounds', biker_time_bounds)

            biker_distance_covered, biker_x_vec, biker_y_vec, biker_min_distance_pass, biker_traj = get_agent_traj(example_traj = biker_traj, check_quality = check_quality, min_distance_bound = min_distance_bound)

            biker_type_pass = biker_type in other_agents

            biker_in_time = get_time_overlap(time_bound_one = motorized_time_bounds, time_bound_two = biker_time_bounds)

            biker_overall_pass = biker_min_distance_pass and biker_type_pass and biker_in_time

            # limit the biker_x_vec to be in the bounds of interest

            if (biker_overall_pass):
                biker_distance_covered, biker_x_vec, biker_y_vec, biker_min_distance_pass, biker_traj = get_agent_traj(example_traj = biker_traj, check_quality = check_quality, min_distance_bound = min_distance_bound, frame_bounds = motorized_time_bounds)
                subset_traj_list.append(biker_traj)
                local_label = '_'.join([biker_type, str(biker_member_id)])
                plt.plot(biker_x_vec, biker_y_vec, label = local_label, ls = '--')
                plt.hold(True)

            # plot speed of agent vs min distance to other agetns

        # plot our agent
        local_label = '_'.join([agent_type, str(member_id)])
        plt.plot(full_x_vec, full_y_vec, label = local_label, lw = 3.0)
        plt.hold(True)

        agent_str = '_'.join(desired_agents + other_agents)
        frame_str = '_'.join([str(x) for x in motorized_time_bounds])
        title_str = ' '.join(['frame: ', frame_str, agent_str])
        plt.title(title_str)

        plot_file = per_agent_plot_dir + '/' + '_'.join(['MotorizedScene', 'frame', frame_str, 'member', str(member_id), 'scene', str(scene_name), 'video', str(video_num), agent_str]) + '.pdf'
        plt.legend()
        plt.savefig(plot_file)
        plt.close()

        # subset traj list
        subset_traj_dict[member_id] = subset_traj_list
        # subset_traj_list[i][k] = agent i, kth frame of trajectory

        subset_traj_output_pkl_path = per_agent_plot_dir + '/agent_' + str(specific_motorized_agent_id) + '_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        # save a pkl of agent of interest
        save_pkl(pkl_path = subset_traj_output_pkl_path, pkl_object = subset_traj_dict)

        # now plot the speed of the motorized agent vs coordinates
        if plot_speed_mode:
            plot_speed_distance_panel(motorized_traj = agent_traj, motorized_agent_id = specific_motorized_agent_id, check_quality = check_quality, scene_name = scene_name, video_num = video_num, plot_results_dir = per_agent_plot_dir, scene = scene, desired_other_agents = ['Biker', 'Pedestrian'], full_x_vec = full_x_vec, full_y_vec = full_y_vec)

    return motorized_time_bounds, example_traj, subset_traj_dict

"""
    given pkl objects of all frames, trajectories per member, plot campus scenes and speeds of agents

"""
if __name__ == "__main__":

    args = parse_args()

    pkl_results_dir = args.pkl_results_dir
    all_plot_results_dir = args.plot_results_dir

    # movies of interest where we have saved pkl
    gates_scene_video_list = ['gates_1', 'gates_6']
    death_scene_video_list = ['deathCircle_0', 'deathCircle_1', 'deathCircle_3']
    scene_video_list = [death_scene_video_list[0]]
    #scene_video_list = gates_scene_video_list + death_scene_video_list
    scene_video_list = [gates_scene_video_list[0]]

    for scene_video in scene_video_list:

        scene_name = scene_video.split('_')[0]
        video_num = int(scene_video.split('_')[1])

        plot_results_dir = all_plot_results_dir + '/' + scene_name + '_' + str(video_num)
        remove_and_create_dir(plot_results_dir)

        # pkl paths for saved dictionaries
        scene_pkl_path = pkl_results_dir + '/frameInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        traj_pkl_path = pkl_results_dir + '/trajInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        frameList_pkl_path = pkl_results_dir + '/frameList_' + scene_name + '_video_' + str(video_num) + '.pkl'

        
        #print('loading scene')
        #scene = load_pkl(pkl_path = scene_pkl_path)
        print('loading member traj')
        member_traj = load_pkl(pkl_path = traj_pkl_path)
        print('loading frame lists')
        all_frame_bounds = load_pkl(pkl_path = frameList_pkl_path)

        # member traj has to be at least this high
        MIN_DISTANCE_COVERED = 300

        check_quality_global = False

        # only plot motorized
        agents = ['Cart', 'Car', 'Bus']
        #agents = ['Bus']

        other_agents = ['Biker', 'Pedestrian']
        #other_agents = ['Biker']

        plot_axes = None

        # to get all the plots at once
        plot_all_motorized_agent(traj_list = member_traj, base_plot_dir = plot_results_dir, scene_name = scene_name, video_num = video_num, desired_agents = agents, other_agents = other_agents, check_quality = check_quality_global, plot_axes = plot_axes, min_distance_bound = MIN_DISTANCE_COVERED, pkl_results_dir = pkl_results_dir)

