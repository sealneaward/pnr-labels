import cPickle as pkl
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

np.seterr(divide='raise', invalid='raise')


def extract_trajectories(annotation_movements, frame_rate=25.0):
    trajectories = []

    for annotation in annotation_movements:
        for player in annotation['players']:
            # create new increasing times for game_clock
            player['movement']['before']['game_clock'] = np.arange(0.0, 2.0, 0.04)
            player['movement']['after']['game_clock'] = np.arange(0.0, 2.0, 0.04)

            before_trajectory = player['movement']['before'][['game_clock', 'x_loc', 'y_loc']].values
            after_trajectory = player['movement']['after'][['game_clock', 'x_loc', 'y_loc']].values

            trajectories.append(before_trajectory)
            trajectories.append(after_trajectory)

    return trajectories


def complete_trajectories(trajectories):
    completed_trajectories = []
    for trajectory in tqdm(trajectories):
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
        
    return completed_trajectories


def generate_behavior_sequences(features_trajectories):
    behavior_sequences = []

    for trajectory_features in tqdm(features_trajectories):
        windows = rolling_window(trajectory_features)
        behavior_sequence = behavior_extract(windows)
        behavior_sequences.append(behavior_sequence)

    return behavior_sequences


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
    # TODO change window length and offset tomorrow(Tuesday)
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
    trajectories = extract_trajectories(action_movements)
    trajectories = complete_trajectories(trajectories)
    features = compute_features(trajectories)
    behaviours = generate_behavior_sequences(features)

    return np.array(behaviours)


if __name__ == '__main__':
    from pnr.data.constant import game_dir

    pnr_dir = os.path.join(game_dir, 'pnr-annotations')
    action_movements = pkl.load(open(os.path.join(pnr_dir, 'roles/trajectories.pkl'), 'rb'))
    behaviours = get_behaviours(action_movements)
    np.save('%s/roles/behaviours' % (pnr_dir), behaviours)

