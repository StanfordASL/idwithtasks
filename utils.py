from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import sys,os
import pickle

from comment_tlmodelid import check_sat

CONTROL_COST = 0.5


def save_pkl(pkl_path = None, pkl_object = None):
    pickle.dump(pkl_object, open(pkl_path, 'wb'))

def remove_and_create_dir(path):
    """ System call to rm -rf and then re-create a dir """
    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.system("mkdir -p " + path)


"""
cost of going left or right is high, 0 is none
"""
def get_control_cost(action = None, control_cost_scalar = CONTROL_COST):

    if( (action == -1) or (action == 1) or (action == 'evade') ):
        cost = control_cost_scalar
    else:
        cost = 0

    return cost


"""
R(b, a, b') = -beta * control_cost(a) + alpha*[H(b) - H(b')]
"""
def compute_reward(B_orig = None, action = None, B_new = None, reward_params_dict = None):

    original_entropy = get_entropy(B_orig)

    new_entropy = get_entropy(B_new)

    entropy_difference = (original_entropy - new_entropy)

    control_cost = get_control_cost(action)

    weighted_control_cost = -reward_params_dict['control_weight']*control_cost

    weighted_info_gain = reward_params_dict['entropy_weight']*entropy_difference

    reward = weighted_control_cost + weighted_info_gain

    info_gain = entropy_difference

    return reward, control_cost, info_gain, weighted_control_cost, weighted_info_gain

"""
R(b, a, b') = -beta * control_cost(a) + alpha*[H(b) - H(b')]
"""
def compute_KL_reward(B_orig = None, action = None, B_new = None, B_true = None, reward_params_dict = None):

    KL = get_KL_divergence(p_vec = B_true, q_vec = B_new)

    #original_entropy = get_entropy(B_orig)
    #new_entropy = get_entropy(B_new)
    #entropy_difference = (original_entropy - new_entropy)

    control_cost = get_control_cost(action)

    weighted_control_cost = -reward_params_dict['control_weight']*control_cost

    weighted_info_gain = reward_params_dict['entropy_weight']*KL

    reward = weighted_control_cost + weighted_info_gain

    info_gain = KL

    return reward, control_cost, info_gain, weighted_control_cost, weighted_info_gain


"""
use bayes' rule to update belief distro
"""
def update_belief(B_orig = None, trans_prob_vector = None):

    B = copy.copy(B_orig)

    B *= trans_prob_vector
    B /= B.sum()
    return B


"""
find prob of a sequence satisfying BLTL spec
1. enumerate all possible reachable state trajectories of horizon h
2. ideally update this to satisfaction probability
3. from here can get a list of all satisfying sequences per model and we can sample from here
4. the probability from here is NOT what we should use to update transitions, rather
use sat_sequences_dict

"""
def get_observation_probability(model_names = None, horizon = None, joint_state_sequences = None):

    num_all_sequences = len(joint_state_sequences)

    # numpy array of trans prob under various models
    p = []

    # dictionary of trans prob per model
    model_prob_dict = {}
    # per model, list of ALL sequences that satisfy model
    sat_sequences_dict = {}

    num_sat_sequences_dict = {}

    for name in model_names:
        sat_sequences = [seq for seq in joint_state_sequences if check_sat(seq,name)]

        num_sat_sequences = len(sat_sequences)

        model_obs_prob = float(num_sat_sequences)/float(num_all_sequences)

        p.append(model_obs_prob)

        sat_sequences_dict[name] = sat_sequences
        num_sat_sequences_dict[name] = num_sat_sequences

        model_prob_dict[name] = model_obs_prob

    return np.array(p), num_sat_sequences_dict, num_all_sequences, sat_sequences_dict, joint_state_sequences , model_prob_dict


"""
joint_state sequence: [[rob, follower], ...]
join with robot state for a joint trajectory that can be model checked
"""
def get_joint_robot_follower_sequences(s_0 = None, max_horizon = None, mode = 'adjacency', print_mode = False):

    init_robot_lane = s_0[0]
    init_follower_lane = s_0[1]

    # starting from init lane, all for the follower car
    # [init, init-1, init, ...]
    all_state_sequences = generate_all_paths(init_lane = init_follower_lane, horizon = max_horizon, mode = mode)

    # joint_state sequence: [[rob, follower], ...]
    # join with robot state for a joint trajectory that can be model checked
    joint_state_sequences = []
    for specific_follower_sequence in all_state_sequences:
        specific_joint_sequence = [s_0]
        for follower_lane in specific_follower_sequence:
            specific_joint_sequence.append([init_robot_lane, follower_lane])
        #print('specific state sequence', specific_joint_sequence)

        # model check the specific joint sequence
        joint_state_sequences.append(specific_joint_sequence)
        if(print_mode):
            for name in ['k-bound', 'pursuer']:
                sat_bool = check_sat(specific_joint_sequence, name)

                print('')
                print('name', name)
                print('sat_bool', sat_bool)
                print(specific_joint_sequence)
                print('')

    return all_state_sequences, joint_state_sequences

"""
find prob of a sequence satisfying BLTL spec
1. enumerate all possible reachable state trajectories of horizon h
2. find those that have a certain [start, stop] pair
    - ideally update this to satisfaction probability
3. THIS IS NOT USED
"""
def get_naive_observation_probability(model_id = None, desired_start_state = None, desired_end_state = None, horizon = None):

    all_state_sequences = generate_all_paths(init_lane = desired_start_state, horizon = horizon, mode = 'adjacency')

    num_all_sequences = len(all_state_sequences)

    # find all sequences that sta
    sat_sequences = [x for x in all_state_sequences if ((x[0] == desired_start_state) and (x[-1] == desired_end_state)) ]

    num_sat_sequences = len(sat_sequences)

    obs_prob = float(num_sat_sequences)/float(num_all_sequences)

    return obs_prob, num_sat_sequences, num_all_sequences, sat_sequences, all_state_sequences


"""
get KL divergence
"""

def get_KL_divergence(p_vec = None, q_vec = None):

    # p: ref distro, true distro
    # q: approximate distro

    KL = 0

    assert(len(p_vec) == len(q_vec))

    for i in range(len(p_vec)):

        p = p_vec[i]
        q = q_vec[i]

        if( (p != 0.0) and (q != 0.0) ):
            KL += -p*np.log( (p)/(q))
    return KL

"""
get entropy of a vector
"""
def get_entropy(vec):
    H = 0

    for x in vec:
        if(x != 0):
            H += -x*np.log(x)

    return H

"""
lane 0 is leftmost, lane MAX_LANES is rightmost
given a lane of interest, get neighbors [left, current, right]
'no_adjacency' mode: debug mode used to check we are enumerating all S^T possible states
use 'adjacency' mode for simulations

"""
def get_neighbors(lane = 3, max_lanes = 4, mode = 'adjacency'):

    if(lane == 0):
        neighbors = [0,1]
    elif(lane == max_lanes):
        neighbors = [max_lanes, max_lanes -1]
    else:
        neighbors = [lane - 1, lane, lane + 1]

    if(mode == 'no_adjacency'):
        neighbors = range(max_lanes + 1)

    return neighbors

"""
get all sequences from 'init_lane' for horizon h
these are observation histories
"""

def generate_all_paths(init_lane = 3, horizon = 3, mode = None):
    s = [init_lane]

    all_state_sequences = {}

    all_state_sequences[0] = [s]

    for h in range(horizon):
        all_state_sequences[h+1] = []


        for state_sequence in all_state_sequences[h]:
            last_state = state_sequence[-1]

            neighbor_list = get_neighbors(last_state, mode = mode)

            for neighbor in  neighbor_list:
                new_state_sequence = list(state_sequence)
                new_state_sequence.append(neighbor)

                all_state_sequences[h+1].append(new_state_sequence)

    return all_state_sequences[h+1]

"""
for a list of sequences concordant with model i, rank each sequence
by how many total models ALSO are concordant

return num_satisfying_model_dict[seq_num] -> [num_sat_models, sequence]
"""

def evaluate_sequence_across_all_models(possible_sequences_for_model = None, model_names = None):
    num_satisfying_model_dict = {}

    # look at all satisfying sequences
    # how many total models satisfied, what is the sequence
    for seq_num, sequence in enumerate(possible_sequences_for_model):
        sat_models = [check_sat(sequence,other_model) for other_model in model_names]

        num_satisfying_model_dict[seq_num] = [sum(sat_models), sequence]

    # get the list of models, with least and most overlap with other models
    num_satisfiers_list = [x[0] for x in num_satisfying_model_dict.values()]

    max_satisfiers = max(num_satisfiers_list)
    min_satisfiers = min(num_satisfiers_list)

    print('max_satisfiers', max_satisfiers)
    print('min_satisfiers', min_satisfiers)

    max_satisfiers_list = []
    min_satisfiers_list = []

    for k,v in num_satisfying_model_dict.iteritems():
        num_satisfiers = v[0]
        spec_seq = v[1]

        if(num_satisfiers == max_satisfiers):
            max_satisfiers_list.append(spec_seq)

        if(num_satisfiers == min_satisfiers):
            min_satisfiers_list.append(spec_seq)

    return num_satisfying_model_dict, max_satisfiers_list, min_satisfiers_list


def random_element_from_list(input_list = None):
    random_index = np.random.choice(range(len(input_list)))
    return input_list[random_index]

def sample_sat_sequence(sat_sequences_dict = None, true_model_name = None, sampling_mode = 'max_overlap', model_names = None):

    # possible sequences
    possible_sequences_for_model = sat_sequences_dict[true_model_name]

    num_satisfying_model_dict, max_satisfiers_list, min_satisfiers_list = evaluate_sequence_across_all_models(possible_sequences_for_model = possible_sequences_for_model, model_names = model_names)


    if(sampling_mode == 'random'):

        sample_model_sequence = random_element_from_list(input_list = possible_sequences_for_model)

    elif(sampling_mode == 'min_overlap'):
        # for each possible state sequence, find one that has least overlap with others

        # get a dict from [state] -> how many congruent models, sample randomly where

        sample_model_sequence = random_element_from_list(input_list = min_satisfiers_list)

    elif(sampling_mode == 'max_overlap'):
        # for each possible state sequence, find one that has least overlap with others
        sample_model_sequence = random_element_from_list(input_list = max_satisfiers_list)

    # deterministic mode
    else:
        sample_model_sequence = possible_sequences_for_model[0]


    return sample_model_sequence

"""
    input: example sequence

    see how many models satisfy that sequence

    sat_p = [1 if seq satisfies model]

    empirical_p = [1/how many total sequences for that model] -> more strict models are upweighted
"""

def get_specific_sequence_prob(model_names = None, example_sequence = None, num_sat_sequences_dict = None, print_mode = True):

    # which models satisfy this sequence
    empirical_p = []
    sat_p = []
    for model in model_names:
        if check_sat(example_sequence, model):
            p = 1.0/float(num_sat_sequences_dict[model])
            empirical_p.append(p)
            sat_p.append(1.0)
        else:

            empirical_p.append(0.0)
            sat_p.append(0.0)

    if print_mode:
        print('empirical_p', empirical_p)
        #print('sat_p', sat_p/sum(sat_p))
        print('sat_p', np.array(sat_p)/sum(sat_p))

    return empirical_p, sat_p

if __name__ == "__main__":

    max_horizon = 6
    init_lane = 2

    true_model_name = 'pursuer'

    # robot, follower
    s_0 = [init_lane -1, init_lane-1]

    possible_init_states = [ [init_lane -1, init_lane],  [init_lane, init_lane], [init_lane + 1, init_lane] ]

    # UNIT TEST: generate all paths
    ##############################################################
    #for mode in ['adjacency', 'no_adjacency']:
    #for mode in ['adjacency']:

    for s_0 in possible_init_states:

        all_state_sequences, joint_state_sequences = get_joint_robot_follower_sequences(s_0 = s_0, max_horizon = max_horizon, mode = 'adjacency', print_mode = False)

        # get all state sequences that match a (start/end) pair
        #obs_prob, num_sat_sequences, num_all_sequences, sat_sequences, all_state_sequences = get_naive_observation_probability(model_id = None, desired_start_state = init_lane, desired_end_state = init_lane, horizon = max_horizon)

        #print('obs prob', obs_prob)
        #print('sat_sequences', sat_sequences)
        #print('all_state_sequences', all_state_sequences)


        # get all state sequences that match a specification
        model_names = ['benign',
                        'k-bound',
                        'pursuer']

        trans_prob_vector, num_sat_sequences_dict, num_all_sequences, sat_sequences_dict, joint_state_sequences, model_prob_dict = get_observation_probability(model_names = model_names, horizon = max_horizon, joint_state_sequences = joint_state_sequences)

        print(' ')
        print(' ')
        print('s_0 [robot, follower]', s_0)
        print('trans_prob_vector', trans_prob_vector)
        for model in model_names:
            print(model, model_prob_dict[model])
            print('true', true_model_name)
            print('num_sat', num_sat_sequences_dict[model])


        # things are correct upto here
        ################################
        num_satisfying_model_dict, max_satisfiers_list, min_satisfiers_list = evaluate_sequence_across_all_models(possible_sequences_for_model = sat_sequences_dict[true_model_name], model_names = model_names)
        print('num max satisfiers', len(max_satisfiers_list))
        print('num min satisfiers', len(min_satisfiers_list))

        # get a sequence that satisfies true model and MOST others and MIN others
        example_sequence = sample_sat_sequence(sat_sequences_dict = sat_sequences_dict, true_model_name = true_model_name, sampling_mode = 'max_overlap', model_names = model_names)
        print('true model sequence', example_sequence, 'max_overlap')
        empirical_p, sat_p = get_specific_sequence_prob(model_names = model_names, example_sequence = example_sequence, num_sat_sequences_dict = num_sat_sequences_dict, print_mode = True)

        example_sequence = sample_sat_sequence(sat_sequences_dict = sat_sequences_dict, true_model_name = true_model_name, sampling_mode = 'min_overlap', model_names = model_names)
        print('true model sequence', example_sequence, 'min_overlap')
        empirical_p, sat_p = get_specific_sequence_prob(model_names = model_names, example_sequence = example_sequence, num_sat_sequences_dict = num_sat_sequences_dict, print_mode = True)
        print(' ')
        print(' ')

        for model in model_names:
            pass
            #example_sequence = sample_sat_sequence(sat_sequences_dict = sat_sequences_dict, true_model_name = model, sampling_mode = 'max_overlap', model_names = model_names)

            #print('model', model)
            #print('example_sequence', example_sequence)


    ## UNIT TEST: rewards
    ###############################################################
    #B_orig = np.array([1/3.,1/3.,1/3.])

    #reward_params_dict = {}
    #reward_params_dict['beta'] = 0.5
    #reward_params_dict['alpha'] = 0.5

    #for action in [-1,0,1]:
    #    # get control cost
    #    cost = get_control_cost(action = action, control_cost_scalar = CONTROL_COST)
    #    print('')
    #    print('action', action)
    #    print('cost', cost)

    #    trans_prob_vector = np.array([1/2.,1/8.,3/8.])

    #    B_new = update_belief(B_orig = B_orig, trans_prob_vector = trans_prob_vector)
    #    print('B_new', B_new)

    #    # get reward
    #    reward, entropy_difference = compute_reward(B_orig = B_orig, action = action, B_new = B_new, reward_params_dict = reward_params_dict)

    #    print('reward', reward)
    #    print('entropy_diff', entropy_difference)
    #    print('')
