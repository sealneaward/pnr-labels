import cPickle as pkl
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing
from copy import copy

np.seterr(divide='raise', invalid='raise')


def convert_movement(movement):
    movement.loc[movement.x_loc > 47, 'y_loc'] = movement.loc[movement.x_loc > 47, 'y_loc'].apply(lambda y: 50 - y)
    movement.loc[movement.x_loc > 47, 'x_loc'] = movement.loc[movement.x_loc > 47, 'x_loc'].apply(lambda x: 94 - x)
    # movement['x_loc'] = movement['y_loc'].apply(lambda y: 250 * (1 - (y - 0) / (50 - 0)) + -250 * ((y - 0) / (50 - 0)))
    # movement['y_loc'] = movement['x_loc'].apply(lambda x: -47.5 * (1 - (x - 0) / (47 - 0)) + 422.5 * ((x - 0) / (47 - 0)))

    return movement


def convert_to_half(annotation_movements):
    for anno_ind, annotation in enumerate(annotation_movements):
        for player_ind, player in enumerate(annotation['players']):
            player['movement']['before'] = convert_movement(player['movement']['before'])
            player['movement']['after'] = convert_movement(player['movement']['after'])
            annotation_movements[anno_ind]['players'][player_ind] = player
    return annotation_movements


def extract_trajectories(annotation_movements):
    trajectories = []
    annotations = []

    for annotation in annotation_movements:
        for player in annotation['players']:
            # create new increasing times for game_clock
            player['movement']['before']['game_clock'] = np.arange(0.0, 2.0, 0.04)
            player['movement']['after']['game_clock'] = np.arange(0.0, 2.0, 0.04)

            before_trajectory = player['movement']['before'][['game_clock', 'x_loc', 'y_loc']].values
            after_trajectory = player['movement']['after'][['game_clock', 'x_loc', 'y_loc']].values

            # append trajectory and annotation
            trajectories.append(before_trajectory)
            trajectories.append(after_trajectory)

            # append annotations for before and after for player
            annotation['annotation']['player_id'] = player['player_id']
            annotation['annotation']['action'] = 'before'
            before_annotation = copy(annotation['annotation'])
            annotations.append(before_annotation)
            annotation['annotation']['action'] = 'after'
            after_annotation = copy(annotation['annotation'])
            annotations.append(after_annotation)


    return trajectories, annotations


def complete_trajectories(trajectories, annotations):
    completed_trajectories = []
    completed_annotations = []

    for ind, trajectory in enumerate(trajectories):
        annotation = annotations[ind]
        completed_trajectory = []
        for i in range(0, len(trajectory)):
            rec = []
            if i == 0:
                # time, location_c, speed_c, rot_c
                rec = [0, 0, 0, 0]
            else:
                loc_c = math.sqrt((trajectory[i][1]-trajectory[i-1][1])**2+(trajectory[i][2]-trajectory[i-1][2])**2)
                rec.append(trajectory[i][0])
                rec.append(loc_c)
                rec.append(loc_c / (trajectory[i][0] - trajectory[i - 1][0]))
                # catch numpy exceptions
                try:
                    rec.append(math.atan((trajectory[i][2]-trajectory[i-1][2]) / (trajectory[i][1]-trajectory[i-1][1])))
                except Exception as err:
                    rec.append(0)
            completed_trajectory.append(rec)
        completed_trajectories.append(completed_trajectory)
        completed_annotations.append(annotation)

    return completed_trajectories, completed_annotations


def generate_behavior_sequences(features_trajectories):
    behavior_sequences = []

    for trajectory_features in tqdm(features_trajectories):
        # shape = 50, 4
        windows = rolling_window(trajectory_features)
        # shape = 5, 10, 4
        behavior_sequence = behavior_extract(windows)
        # shape = 5, 18
        behavior_sequences.append(behavior_sequence)

    return behavior_sequences


def generate_normal_behavior_sequence(behavior_sequences):
    behavior_sequences_normal = []
    templist = []
    for item in behavior_sequences:
        for ii in item:
            templist.append(ii)
        print len(item)
    print len(templist)
    min_max_scaler = preprocessing.MinMaxScaler()
    # print np.shape(behavior_sequence)
    templist_normal = min_max_scaler.fit_transform(templist).tolist()
    index = 0
    for item in behavior_sequences:
        behavior_sequence_normal = []
        for ii in item:
            behavior_sequence_normal.append(templist_normal[index])
            index = index + 1
        behavior_sequences_normal.append(behavior_sequence_normal)

    return behavior_sequences_normal

def compute_features(completed_trajectories):
    features_trajectories = []
    for trajectory in tqdm(completed_trajectories):
        trajectory_features = []
        for i in range(0,len(trajectory)):
            rec = []
            if i == 0:
                # time, locationC, speedC, rotC
                rec = [0, 0, 0, 0]
            else:
                loc_c = trajectory[i][1]
                loc_c_rate = loc_c / (trajectory[i][0] - trajectory[i-1][0])
                rec.append(trajectory[i][0])
                rec.append(loc_c_rate)
                # TODO check removal of if for loc_c_rate
                rec.append(trajectory[i][2]-trajectory[i-1][2])
                rec.append(trajectory[i][3]-trajectory[i-1][3])
            trajectory_features.append(rec)
        features_trajectories.append(trajectory_features)
        
    return features_trajectories


def rolling_window(sample, window_size=10, offset=5):
    # TODO change window length and offset
    time_length = len(sample) # should be around 50
    window_length = int(time_length / window_size)
    windows = []
    for i in range(0, window_length):
        windows.append([])

    for ind, record in enumerate(sample):
        time = ind
        for i in range(0, window_length):
            if (time > (i * offset)) & (time <= (i * offset + window_size)):
                windows[i].append(record)
    return windows


def behavior_extract(windows):
    behavior_sequence = []
    for window in windows:
        behaviour_feature = []
        records = np.array(window)
        if len(records) != 0:

            data = pd.DataFrame(records)
            description = data.describe()
            skip_these = [0, 2]

            for i in range(8):
                if i in skip_these:
                    continue

                behaviour_feature.append(description[1][i])
                behaviour_feature.append(description[2][i])
                behaviour_feature.append(description[3][i])

            behavior_sequence.append(behaviour_feature)

    return behavior_sequence


def get_behaviours(action_movements):
    """
    Use feature extraction methods described in
    "Yao, D., Zhang, C., Zhu, Z., Huang, J., & Bi, J. (2017, May).
    Trajectory clustering via deep representation learning.
    In Neural Networks (IJCNN), 2017 International Joint Conference on (pp. 3880-3887)."

    Parameters
    ----------
    action_movements: dict
        before and after action movement information for annotation

    Returns
    -------
    behaviours: np.array
        behaviour vectors for each action identified
    """
    # TODO cleanup
    # action_movements = convert_to_half(action_movements)
    trajectories, annotations = extract_trajectories(action_movements)
    trajectories, annotations = complete_trajectories(trajectories, annotations)
    features = compute_features(trajectories)
    behaviours = generate_behavior_sequences(features)
    # behaviours = generate_normal_behavior_sequence(behaviours)

    return np.array(behaviours), annotations


if __name__ == '__main__':
    from pnr.data.constant import game_dir

    pnr_dir = os.path.join(game_dir, 'pnr-annotations')
    action_movements = pkl.load(open(os.path.join(pnr_dir, 'roles/trajectories.pkl'), 'rb'))
    behaviours, annotations = get_behaviours(action_movements)

    np.save('%s/roles/behaviours' % (pnr_dir), behaviours)
    pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/annotations.pkl'), 'wb'))

