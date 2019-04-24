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

from plotting_utils import *
from RL_utils import *
from parse_drone_data import *



def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def parse_args():

    data_dir_default = '~/idwithtasks/RL/drone_data/'
    plot_dir_default = '~/idwithtasks/RL/drone_data/plots/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_dir', type=str, required=False, default = data_dir_default,  help="")
    parser.add_argument('--base_plot_dir', type=str, required=False, default = plot_dir_default,  help="")

    args = parser.parse_args()
    return args

def get_car_frames(annotations_file = None, base_results_dir = None, scene_name = None, video_num = None, agents_list = ['Cart', 'Car', 'Bus']):
    interesting_agents = {}

    # interesting_agents['Cart'] = [[5,10], [12, 19], [25, 30]]

    member_index = 0
    xmin_index = 1
    ymin_index = 2
    xmax_index = 3
    ymax_index = 4

    frame_index = 5
    lost_index = 6
    occluded_index = 7
    generated_index = 8
    label_index = 9

    with open(annotations_file, 'r') as f:
        for line in f:

            split_line = line.split('\n')[0].split()
            member_id = split_line[member_index]

            xmin  = split_line[xmin_index]
            ymin  = split_line[ymin_index]

            xmax  = split_line[xmax_index]
            ymax  = split_line[ymax_index]

            frame = split_line[frame_index]

            # add a check to only look at frames of interest
            lost = split_line[lost_index]
            occluded = split_line[occluded_index]
            generated = split_line[generated_index]

            class_label = re.split(r'"', split_line[label_index])[1]

            if(class_label in agents_list):
                if(class_label in interesting_agents.keys()):
                    frame_bounds_per_class = interesting_agents[class_label]
                    frame_bounds_per_class.append(int(frame))
                    interesting_agents[class_label] = frame_bounds_per_class

                else:
                    interesting_agents[class_label] = [int(frame)]

    frame_bounds_dict = {}
    all_bounds = []
    for agent, frame_list in interesting_agents.iteritems():
        car_frames = list(set(frame_list))
        car_frames.sort()

        frame_bounds_list = []
        last_frame = -2

        for i, frame in enumerate(car_frames):

            # if next frame, don't add
            if frame == (last_frame + 1):
                pass
            else:
                if(last_frame > 0):
                    frame_bounds_list.append(last_frame)
                frame_bounds_list.append(frame)
            last_frame = frame
        frame_bounds_list.append(car_frames[-1])

        frame_bounds = []
        for i, x in enumerate(frame_bounds_list):
            if(i % 2 == 0):
                start_end_bounds = [x, frame_bounds_list[i+1]]
                frame_bounds.append(start_end_bounds)
                all_bounds.append(start_end_bounds)

        frame_bounds_dict[agent] = frame_bounds

    return interesting_agents, frame_bounds_dict, all_bounds


def save_scene_info_pkl(annotations_file = None, base_plot_dir = None, scene_name = None, video_num = None, all_bounds = None):
    scene = {}
    member_traj = {}
    # member_traj[member_id] = {type: 'Cart', xy_list = [(x,y, frame, occluded_str), ...], scene: video_num}

    member_index = 0
    xmin_index = 1
    ymin_index = 2
    xmax_index = 3
    ymax_index = 4

    frame_index = 5
    lost_index = 6
    occluded_index = 7
    generated_index = 8
    label_index = 9

    with open(annotations_file, 'r') as f:
        for line in f:

            split_line = line.split('\n')[0].split()
            member_id = split_line[member_index]

            xmin  = split_line[xmin_index]
            ymin  = split_line[ymin_index]

            xmax  = split_line[xmax_index]
            ymax  = split_line[ymax_index]

            frame = int(split_line[frame_index])

            # add a check to only look at frames of interest
            lost = split_line[lost_index]
            occluded = split_line[occluded_index]
            generated = split_line[generated_index]

            class_label = re.split(r'"', split_line[label_index])[1]

            x = 0.5*(float(xmin) + float(xmax))
            y = 0.5*(float(ymin) + float(ymax))

            quality = '_'.join([lost, occluded])

            object_dict = {'member_id':  member_id, 'xy': (x,y) , 'class': class_label, 'frame': frame, 'lost': lost, 'occluded': occluded, 'generated': generated, 'quality': quality}

            necessary_frame = False
            if all_bounds:
                for bounds in all_bounds:
                    min_bound = bounds[0]
                    max_bound = bounds[1]

                    if (frame >= min_bound and frame <= max_bound):
                        necessary_frame = True
            else:
                necessary_frame = True

            if necessary_frame:
                if(frame in scene.keys()):

                    scene[frame].append(object_dict)
                else:
                    scene[frame] = [object_dict]

                if member_id in member_traj.keys():
                    member_traj[member_id].append(object_dict)
                else:
                    member_traj[member_id] = [object_dict]


    # write all scene info to a file
    #######
    scene_pickle_path = base_plot_dir + '/frameInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
    pickle.dump(scene, open(scene_pickle_path, 'wb'))

    traj_pickle_path = base_plot_dir + '/trajInfo_scene_' + scene_name + '_video_' + str(video_num) + '.pkl'
    pickle.dump(member_traj, open(traj_pickle_path, 'wb'))

    return scene, member_traj


def compute_and_plot_agent_speed(member_traj = None, base_plot_dir = None, scene_name = None, video_num = None, boxplot_ylim_vector = None):

    member_speed_df = pandas.DataFrame()
    # per member, get a distro of the speeds

    for member_id, member_traj_list in member_traj.iteritems():

        speed_vec = get_agent_velocity_vector(member_traj_list = member_traj_list, check_quality = True)

        local_df = pandas.DataFrame()

        local_df['speed'] = speed_vec

        local_df['agent_type'] = [member_traj_list[0]['class'] for x in range(len(speed_vec))]

        local_df['member_id'] = [member_traj_list[0]['member_id'] for x in range(len(speed_vec))]

        # [{'quality': '0_0', 'frame': '0', 'generated': '0', 'xy': (888.0, 1162.5), 'member_id': '24', 'class': 'Biker'}, ...]
        member_speed_df = member_speed_df.append(local_df)

    # boxplot of speed vs class
    plot_file = base_plot_dir + '/boxplot_speed_scene_' + scene_name + '_video_' + str(video_num) + '.pdf'
    plot_grouped_boxplot(df = member_speed_df, x_var = 'agent_type', y_var = 'speed', plot_file = plot_file, ylim = boxplot_ylim_vector)

    return member_speed_df

def compute_intraAgent_distance(scene = None, base_plot_dir = None, scene_name = None, video_num = None):

    # per frame, get a list of all interactions, and all multi agent dicts
    all_interactions_across_video = []
    all_multiAgent_per_video = []

    all_possible_agent_interactions = set([])

    for frame, example_scene in scene.iteritems():

        # all pairwise interactions
        all_interaction_list = get_all_interactions_frame(single_frame_scene_list = example_scene, frame = frame, scene_name = scene_name, good_images_only = False)

        agent_type_list = set([x['type'] for x in all_interaction_list])

        all_possible_agent_interactions = all_possible_agent_interactions.union(agent_type_list)

        all_interactions_across_video.append(all_interaction_list)

        local_desired_objects = [[x.split('_')[0], x.split('_')[1]] for x in list(all_possible_agent_interactions)]

        multi_agent_dict = get_distro_agent_interactions(all_interaction_list = all_interaction_list, single_frame_scene_list = example_scene, frame = frame, scene_name = scene_name, desired_objects = local_desired_objects, good_images_only = True)

        all_multiAgent_per_video.append(multi_agent_dict)

    print(all_possible_agent_interactions)
    # now have the all_multiAgent dict
    # set(['Car_Pedestrian', 'Bus_Car', 'Biker_Car', 'Pedestrian_Pedestrian', 'Bus_Pedestrian', 'Car_Car', 'Biker_Bus', 'Biker_Pedestrian', 'Biker_Biker'])

    desired_objects = [[x.split('_')[0], x.split('_')[1]] for x in all_possible_agent_interactions]

    # plot a CDF of min, max, median vector across time
    # get a pandas dataframe

    ############################################################
    agent_distance_df = pandas.DataFrame()

    for agent_interaction in desired_objects:

        interaction_type = '_'.join(agent_interaction)

        min_dist_vec = [x[interaction_type]['min_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]

        max_dist_vec = [x[interaction_type]['max_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]

        num_dist_vec = [x[interaction_type]['num_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]

        type_list = [interaction_type for x in range(len(min_dist_vec))]

        local_df = pandas.DataFrame()
        local_df['agent_type'] = type_list
        local_df['min_dist'] = min_dist_vec
        local_df['max_dist'] = max_dist_vec
        local_df['num'] = num_dist_vec

        agent_distance_df = agent_distance_df.append(local_df)

    return agent_distance_df, desired_objects, all_interactions_across_video, all_multiAgent_per_video, all_possible_agent_interactions


def plot_intraAgent_distance(full_agent_distance_df = None, base_plot_dir = None, desired_objects = None, all_possible_agent_interactions = None):

    chunked_interactions = chunker(list(all_possible_agent_interactions), 3)

    for i, subplot_interactions in enumerate(chunked_interactions):

        print(subplot_interactions)

        agent_distance_df = full_agent_distance_df[full_agent_distance_df['agent_type'].isin(subplot_interactions)]

        # get a boxplot of distance df
        plot_file = base_plot_dir + '/min_distBoxplot_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        plot_grouped_boxplot(df = agent_distance_df, x_var = 'agent_type', y_var = 'min_dist', plot_file = plot_file)

        plot_file = base_plot_dir + '/max_distBoxplot_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        plot_grouped_boxplot(df = agent_distance_df, x_var = 'agent_type', y_var = 'max_dist', plot_file = plot_file)

        plot_file = base_plot_dir + '/numAgent_distBoxplot_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        plot_grouped_boxplot(df = agent_distance_df, x_var = 'agent_type', y_var = 'num', plot_file = plot_file)

        # now do a couple cdfs
        ############################################################

        legend_vec = []
        min_dist_list = []
        max_dist_list = []
        num_dist_list = []

        desired_objects = [[x.split('_')[0], x.split('_')[1]] for x in subplot_interactions]

        for agent_interaction in desired_objects:

            interaction_type = '_'.join(agent_interaction)

            min_dist_vec = [x[interaction_type]['min_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]

            min_dist_list.append(min_dist_vec)

            max_dist_vec = [x[interaction_type]['max_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]
            max_dist_list.append(max_dist_vec)

            num_dist_vec = [x[interaction_type]['num_dist'] for x in all_multiAgent_per_video if interaction_type in x.keys()]
            num_dist_list.append(num_dist_vec)

            legend_vec.append(interaction_type)

        plot_file = base_plot_dir + '/min_distPDF_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        xlabel = 'min distance [unitless]'
        plot_several_cdf(data_vector_list = min_dist_list, xlabel = xlabel, plot_file = plot_file, title_str = scene_name, legend = legend_vec)

        plot_file = base_plot_dir + '/max_distPDF_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        xlabel = 'max distance [unitless]'
        plot_several_cdf(data_vector_list = max_dist_list, xlabel = xlabel, plot_file = plot_file, title_str = scene_name, legend = legend_vec)

        plot_file = base_plot_dir + '/num_distPDF_' + scene_name + '_video_' + str(video_num) + '.' + str(i) + '.pdf'
        xlabel = 'num agents'
        plot_several_cdf(data_vector_list = num_dist_list, xlabel = xlabel, plot_file = plot_file, title_str = scene_name, legend = legend_vec)

def plot_example_traj_2d(example_traj = None, base_plot_dir = None, scene_name = None, video_num = None):

    agent_type = example_traj[0]['class']
    member_id = example_traj[0]['member_id']

    x_vec = [x['xy'][0] for x in example_traj if x['quality'] == '0_0']

    y_vec = [x['xy'][1] for x in example_traj if x['quality'] == '0_0']

    plt.plot(x_vec, y_vec)

    plot_file = base_plot_dir + '/' + '_'.join(['traj', 'member', member_id, 'agent', agent_type, 'scene', str(scene_name), 'video', str(video_num)]) + '.pdf'

    plt.savefig(plot_file)
    plt.close()

if __name__ == "__main__":

    args = parse_args()

    base_data_dir = args.base_data_dir

    base_plot_dir = args.base_plot_dir

    #scene_name = 'deathCircle'
    #video_num = 3

    #scene_video_list = ['gates_1', 'gates_6', 'gates_8']
    scene_video_list = ['gates_1']
    scene_video_list = ['deathCircle_0','deathCircle_1', 'deathCircle_3']

    for scene_video in scene_video_list:

        scene_name = scene_video.split('_')[0]
        video_num = int(scene_video.split('_')[1])

        annotations_file = base_data_dir + '/' + scene_name + '/video' + str(video_num) + '/annotations.txt'

        interesting_agents, frame_bounds_dict, all_bounds = get_car_frames(annotations_file = annotations_file, base_results_dir = base_plot_dir, scene_name = scene_name, video_num = video_num)

        # pkl: scene, member_traj, all_bounds
        # pkl the list of all frames of interest

        frame_list_pkl_path  = base_plot_dir + '/frameList_' + scene_name + '_video_' + str(video_num) + '.pkl'
        pickle.dump(all_bounds, open(frame_list_pkl_path, 'wb'))

        for agent, frame_bounds in frame_bounds_dict.iteritems():
            print(agent, frame_bounds)

        ## parse annotations for distance info
        scene, member_traj = save_scene_info_pkl(annotations_file = annotations_file, base_plot_dir = base_plot_dir, scene_name = scene_name, video_num = video_num, all_bounds = all_bounds)

        ## plot the speed per agent
        boxplot_ylim_vector = [0, 20]

        compute_and_plot_agent_speed(member_traj = member_traj, base_plot_dir = base_plot_dir, scene_name = scene_name, video_num = video_num, boxplot_ylim_vector = boxplot_ylim_vector)

        # intra agent distance
#        agent_distance_df, desired_objects, all_interactions_across_video, all_multiAgent_per_video, all_possible_agent_interactions = compute_intraAgent_distance(scene = scene, base_plot_dir = base_plot_dir, scene_name = scene_name, video_num = video_num)
#
#        plot_intraAgent_distance(full_agent_distance_df = agent_distance_df, base_plot_dir = base_plot_dir, desired_objects = desired_objects, all_possible_agent_interactions = all_possible_agent_interactions)
#
        #for member_id, example_traj in member_traj.iteritems():

        #    plot_items = ['Cart', 'Biker', 'Bus', 'Car']

        #    if(example_traj[0]['class'] in plot_items):
        #        plot_example_traj_2d(example_traj = example_traj, base_plot_dir = base_plot_dir, scene_name = scene_name, video_num = video_num)
