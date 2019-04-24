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

def parse_args():
    plot_results_dir = '~/idwithtasks/RL/drone_data/plots/scenes/'
    pkl_results_dir = '~/idwithtasks/RL/drone_data/plots/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_results_dir', type=str, required=False, default = plot_results_dir,  help="")
    parser.add_argument('--pkl_results_dir', type=str, required=False, default = pkl_results_dir,  help="")

    args = parser.parse_args()
    return args


def load_pkl(pkl_path = None):

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

# list of trajectories, find the bounds
def plot_campus_scene(traj_list = None, base_plot_dir = None, scene_name = None, video_num = None, frame_bounds = None, desired_agents = None, traj_len_cutoff = None, color_dict = None, by_member = False, interpolate_traj = True, TRAJ_INTERPOLATE_LEN = 10, check_quality = False):

    speed_info = []

    for member_id, example_traj in traj_list.iteritems():
        agent_type = example_traj[0]['class']
        member_id = example_traj[0]['member_id']
        frame = int(example_traj[0]['frame'])

        if ((frame >= frame_bounds[0]) and (frame <= frame_bounds[1])):

            if check_quality:
                full_x_vec = [x['xy'][0] for x in example_traj if x['quality'] == '0_0']
                full_y_vec = [x['xy'][1] for x in example_traj if x['quality'] == '0_0']

            else:
                full_x_vec = [x['xy'][0] for x in example_traj]
                full_y_vec = [x['xy'][1] for x in example_traj]


            num_pts = len(full_x_vec)/TRAJ_INTERPOLATE_LEN

            if interpolate_traj:
                x_vec = [full_x_vec[TRAJ_INTERPOLATE_LEN*i] for i in range(num_pts)]
                y_vec = [full_y_vec[TRAJ_INTERPOLATE_LEN*i] for i in range(num_pts)]

            else:
                x_vec = full_x_vec
                y_vec = full_y_vec

            if ( (agent_type in desired_agents) and  (len(x_vec) >= traj_len_cutoff) ):

                if by_member:
                    local_label = '_'.join([agent_type, str(member_id)])
                    plt.plot(x_vec, y_vec, label = local_label)
                else:
                    local_label = agent_type
                    plt.plot(x_vec, y_vec, color=color_dict[agent_type], label = local_label)
                plt.hold(True)

                # now get the speed for this agent
                speed_vec = get_agent_velocity_vector(member_traj_list = example_traj, check_quality = check_quality)
                # speed_vec, min_distance_vec = plot_speed_vs_distance(example_traj = example_traj, scene_dict = scene_dict, scene_name = scene_name)

                speed_dict = {'member_id': member_id, 'agent_type': agent_type, 'frame': frame, 'speed_vec': speed_vec}
                speed_info.append(speed_dict)

                # plot speed of agent vs min distance to other agetns


    agent_str = '_'.join(desired_agents)
    frame_str = '_'.join([str(x) for x in frame_bounds])
    title_str = ' '.join(['frame: ', frame_str, agent_str])
    plt.title(title_str)

    plot_file = base_plot_dir + '/' + '_'.join(['CampusScene', 'frame', str(frame_bounds[0]), str(frame_bounds[1]), 'scene', str(scene_name), 'video', str(video_num), agent_str]) + '.pdf'

    if not by_member:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    else:
        plt.legend()

    plt.savefig(plot_file)
    plt.close()

    return speed_info


def get_min_distance_per_frame(current_scene = None, current_frame = None, scene_name = None, member_id = None, desired_other_agents = None):

    # get minimum interaction distance per frame
    all_interaction_list = get_all_interactions_frame(single_frame_scene_list = current_scene, frame = current_frame, scene_name = scene_name, good_images_only = True)

    # an example interaction
    # {'distance': 280.0803456153252, 'lost': '0_0', 'occluded': '0_0', 'frame': 0, 'member_type': '125_154', 'scene_name': 'deathCircle', 'type': 'Biker_Biker'}

    possible_interactions = []
    for interactions in all_interaction_list:
        agent0 = interactions['type'].split('_')[0]
        agent1 = interactions['type'].split('_')[1]

        member0 = interactions['member_type'].split('_')[0]
        member1 = interactions['member_type'].split('_')[1]

        # if the interaction involves our agent of interest (member_id) and other agent is
        if ( (member0 == member_id) or (member1 == member_id) ):
            # if one of agents is a biker,pedestrian agent
            if ( (agent0 in desired_other_agents) or (agent1 in desired_other_agents) ):
                possible_interactions.append(interactions['distance'])

    if len(possible_interactions) > 0:
        min_distance = np.min(possible_interactions)
    else:
        min_distance = np.nan

    return min_distance

# plot the smoothed speed of the agent, plot the minimum distance to a biker or pedestrian alongside it, normalize both to zero 1
def plot_speed_vs_distance(example_traj = None, scene_dict = None, scene_name = None, desired_other_agents = ['Biker', 'Pedestrian'], check_quality = False):

    agent_type = example_traj[0]['class']
    member_id = example_traj[0]['member_id']
    frame_list = np.array([x['frame'] for x in example_traj])

    min_distance_vec = []
    speed_vec = []
    for i in range(len(example_traj)-1):

        current_coordinate = example_traj[i]['xy']
        next_coordinate = example_traj[i+1]['xy']

        current_frame = int(example_traj[i]['frame'])
        next_frame = int(example_traj[i+1]['frame'])

        current_scene = scene_dict[current_frame]

        min_distance = get_min_distance_per_frame(current_scene = current_scene, current_frame = current_frame, scene_name = scene_name, member_id = member_id, desired_other_agents = desired_other_agents)

        if(next_frame == (current_frame + 1) ):
            xdot = next_coordinate[0] - current_coordinate[0]
            ydot = next_coordinate[1] - current_coordinate[1]

            speed = ((xdot)**2 + (ydot)**2)**(0.5)

        quality_met = (example_traj[i]['quality'] == '0_0') and (example_traj[i+1]['quality'] == '0_0')

        if(check_quality):
            if(quality_met):
                speed_vec.append(speed)
                min_distance_vec.append(min_distance)
        else:
            speed_vec.append(speed)
            min_distance_vec.append(min_distance)

    return speed_vec, min_distance_vec




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
    #scene_video_list = [death_scene_video_list[-1]]
    scene_video_list = death_scene_video_list

    for scene_video in scene_video_list:

        scene_name = scene_video.split('_')[0]
        video_num = int(scene_video.split('_')[1])

        plot_results_dir = all_plot_results_dir + '/' + scene_name + '_' + str(video_num)
        remove_and_create_dir(plot_results_dir)

        # pkl paths for saved dictionaries
        scene_pkl_path = pkl_results_dir + '/frameInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        traj_pkl_path = pkl_results_dir + '/trajInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
        frameList_pkl_path = pkl_results_dir + '/frameList_' + scene_name + '_video_' + str(video_num) + '.pkl'

        speed_file = plot_results_dir + '/speed_' + scene_name + '_video_' + str(video_num) + '.txt'

        print('loading scene')
        scene = load_pkl(pkl_path = scene_pkl_path)
        print('loading member traj')
        member_traj = load_pkl(pkl_path = traj_pkl_path)
        print('loading frame lists')
        all_frame_bounds = load_pkl(pkl_path = frameList_pkl_path)

        # just to inspect dictionary structure
        example_frame = scene.keys()[0]
        example_scene = scene[example_frame]

        # member traj has to be at least this high
        TRAJ_LEN_CUTOFF = 50

        check_quality_global = False

        # all_frame_bounds = [[3000, 3531], [4000, 4531], [5500, 6031], [7000, 7531], [12500, 12720]]
        color_dict = {'Cart': 'green', 'Biker': 'purple', 'Bus': 'red', 'Pedestrian': 'blue', 'Car': 'yellow'}

        with open(speed_file, 'w') as f:

            # quantiles of speed per member
            header_str = '\t'.join(['frame', 'agent_type', 'member_id', 'q1', 'q10', 'q25', 'q50', 'q75', 'q90', 'q99'])
            f.write(header_str + '\n')

            # look at all [frame_start, frame_end] where we have a motorized agent
            for frame_bounds in all_frame_bounds:
                agents = ['Cart', 'Biker', 'Pedestrian', 'Car', 'Bus']

                # plot all above agents trajectories for full frame sequence
                plot_campus_scene(traj_list = member_traj, base_plot_dir = plot_results_dir, scene_name = scene_name, video_num = video_num, frame_bounds = frame_bounds, desired_agents = agents, traj_len_cutoff = TRAJ_LEN_CUTOFF, color_dict = color_dict, interpolate_traj = False, TRAJ_INTERPOLATE_LEN = 1, check_quality = check_quality_global)

                # only plot motorized
                agents = ['Cart', 'Car', 'Bus']
                speed_info = plot_campus_scene(traj_list = member_traj, base_plot_dir = plot_results_dir, scene_name = scene_name, video_num = video_num, frame_bounds = frame_bounds, desired_agents = agents, traj_len_cutoff = TRAJ_LEN_CUTOFF, color_dict = color_dict, by_member = True, interpolate_traj = False, TRAJ_INTERPOLATE_LEN = 1, check_quality = check_quality_global)

                # for cars, get the quantiles of speed, plot speed as a function of distance to other agents
                for speed_element in speed_info:
                    out_vec = [str(speed_element['frame']), speed_element['agent_type'], speed_element['member_id']]

                    speed_ts = np.array(speed_element['speed_vec'])

                    if len(speed_ts) > 0:
                        for percentile in [1, 10, 25, 50, 75, 90, 99]:
                            distro_value = np.percentile(speed_ts, percentile)
                            out_vec.append(str(distro_value))

                        out_str = '\t'.join(out_vec)
                        f.write(out_str + '\n')

                        title_str = '_'.join([speed_element['agent_type'], speed_element['member_id']])

                        speed_plot_file = plot_results_dir + '/plotSpeed_' + scene_name + '_video_' + str(video_num) + title_str + '.pdf'
                        overlaid_plot_file = plot_results_dir + '/overlaidSpeed_' + scene_name + '_video_' + str(video_num) + title_str + '.pdf'

                        if(np.mean(speed_ts) > 0.0):
                            basic_plot_ts(ts_vector = speed_ts, title_str = title_str, plot_file = speed_plot_file, ylabel = 'speed', lw=3.0)

                            example_traj =  member_traj[speed_element['member_id']]
                            speed_vec, min_distance_vec = plot_speed_vs_distance(example_traj = example_traj, scene_dict = scene, scene_name = scene_name, desired_other_agents = ['Biker', 'Pedestrian'], check_quality = check_quality_global)

                            speed_vs_distance_dict = {}
                            speed_vs_distance_dict['norm_speed'] = speed_vec/np.max(speed_vec)
                            speed_vs_distance_dict['norm_distance'] = min_distance_vec/np.max(min_distance_vec)

                            overlaid_ts(normalized_ts_dict = speed_vs_distance_dict, title_str = title_str, plot_file = overlaid_plot_file, ylabel = 'norm values')
