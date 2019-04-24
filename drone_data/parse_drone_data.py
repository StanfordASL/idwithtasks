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

"""
    generate a ts of number of agents of specific type per frame
    this is a proxy for the density of traffic across time

    region_bounds = [x_bound, y_bound]
        - x_bound = [x_min, x_max]

    scene: scene dict

    frame_bounds: list of frames to care about = [500, 1000]

"""

def estimate_traffic_density(region_bounds = None, desired_agents = ['Biker', 'Pedestrian'], scene = None, frame_bounds = None, print_mode = False):

    x_bound = region_bounds[0]
    y_bound = region_bounds[1]

    density_ts = []

    for frame, scene_info in scene.iteritems():

        member_list = []

        # check if we care about this frame
        if ((frame >= frame_bounds[0]) and (frame <= frame_bounds[1])):

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

    return density_ts


def parse_args():

    data_dir_default = '~/idwithtasks/RL/drone_data/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_dir', type=str, required=False, default = data_dir_default,  help="")

    args = parser.parse_args()
    return args

def calc_distance(xy_one = None, xy_two = None):
    return ((xy_one[0] - xy_two[0])**2 + (xy_one[1] - xy_two[1])**2)**(0.5)


def get_single_interaction(object_dict_one = None, object_dict_two = None, scene_name = None, frame = None):
    interaction = {}

    # make sure 'Car, Pedestrian' and 'Pedestrian, Car' don't both exist, have to sort alphabetically
    interaction_type = '_'.join([min(object_dict_one['class'], object_dict_two['class']), max(object_dict_one['class'], object_dict_two['class'])])

    member_type = object_dict_one['member_id'] + '_' + object_dict_two['member_id']

    dist = calc_distance(xy_one = object_dict_one['xy'], xy_two = object_dict_two['xy'])

    lost = object_dict_one['lost'] + '_' + object_dict_two['lost']

    occluded = object_dict_one['occluded'] + '_' + object_dict_two['occluded']

    interaction['type'] = interaction_type
    interaction['scene_name'] = scene_name
    interaction['frame'] = frame
    interaction['member_type'] = member_type
    interaction['distance'] = dist
    interaction['lost'] = lost
    interaction['occluded'] = occluded

    # smaller_list = [x for x in example_scene if (x['lost'] == '0' and x['occluded'] == '0')]
    return interaction


def get_all_interactions_frame(single_frame_scene_list = None, frame = None, scene_name = None, good_images_only = True):
    # prune based on whether images are occluded
    if good_images_only:
        smaller_scene_list = [x for x in single_frame_scene_list if (x['lost'] == '0' and x['occluded'] == '0')]
    else:
        smaller_scene_list = single_frame_scene_list

    # get all pairwise combinations
    all_agent_pairs_list = [y for y in itertools.combinations(smaller_scene_list, 2)]

    all_interaction_list = [get_single_interaction(object_dict_one = x[0], object_dict_two = x[1], scene_name = scene_name, frame = frame) for x in all_agent_pairs_list]

    return all_interaction_list

def get_distro_agent_interactions(all_interaction_list = None, single_frame_scene_list = None, frame = None, scene_name = None, desired_objects = [['Cart', 'Biker'], ['Cart', 'Cart']], good_images_only = True):

    # get all the agents I care about

    # result: 'Cart-Cart': [min_dist, max_dist, median_dist, num_total]

    # result: 'Cart-Biker': [min_dist, max_dist, median_dist, num_total]

    multi_agent_dict = {}
    multi_agent_dict['frame'] = frame
    multi_agent_dict['scene_name'] = scene_name

    for agent_interaction in desired_objects:

        agent_0 = agent_interaction[0] # Cart
        agent_1 = agent_interaction[1] # Biker

        multi_agent_type = '_'.join([agent_0, agent_1])
        multi_agent_type_rev = '_'.join([agent_1, agent_0])

        # parse thru interactions that match agents and pass image quality filter
        # {'distance': 42.24038352098617, 'lost': '1_1', 'occluded': '0_0', 'frame': '344', 'member_type': '0_1', 'scene_name': 'deathCircle', 'type': 'Cart_Cart'}
        if good_images_only:
            high_quality_interactions = [x for x in all_interaction_list if (x['lost'] == '0_0' and x['occluded'] == '0_0')]
        else:
            high_quality_interactions = all_interaction_list

        distance_vec = np.array([x['distance'] for x in high_quality_interactions if ((x['type'] == multi_agent_type) or (x['type'] == multi_agent_type_rev))])

        member_ids = [x['member_type'] for x in high_quality_interactions if ((x['type'] == multi_agent_type) or (x['type'] == multi_agent_type_rev)) ]

        num_member_ids = len(list(set(member_ids)))
        num_distances = len(distance_vec)

        if(num_member_ids > 0 and num_distances > 0):
            min_dist = np.min(distance_vec)
            max_dist = np.max(distance_vec)
            median_dist = np.median(distance_vec)


            #print(member_ids)
            #print('num_members', num_member_ids)
            #print('num_distances', num_distances)

            multi_agent_dict[multi_agent_type] = {'min_dist': min_dist, 'max_dist': max_dist, 'median_dist': median_dist, 'num_members': num_member_ids, 'num_dist': num_distances}


    return multi_agent_dict


def get_agent_velocity_vector(member_traj_list = None, check_quality = True):

    # [{'quality': '0_0', 'frame': '0', 'generated': '0', 'xy': (888.0, 1162.5), 'member_id': '24', 'class': 'Biker'}, ...]
    # TODO: check quality of data

    speed_vec = []
    for i in range(len(member_traj_list)-1):

        current_coordinate = member_traj_list[i]['xy']
        next_coordinate = member_traj_list[i+1]['xy']

        current_frame = int(member_traj_list[i]['frame'])
        next_frame = int(member_traj_list[i+1]['frame'])

        if(next_frame == (current_frame + 1) ):
            xdot = next_coordinate[0] - current_coordinate[0]
            ydot = next_coordinate[1] - current_coordinate[1]

            speed = ((xdot)**2 + (ydot)**2)**(0.5)

        quality_met = (member_traj_list[i]['quality'] == '0_0') and (member_traj_list[i+1]['quality'] == '0_0')

        if(check_quality):
            if(quality_met):
                speed_vec.append(speed)
        else:
            speed_vec.append(speed)

    return speed_vec
