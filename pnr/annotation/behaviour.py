import cPickle as pkl
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing
from copy import copy
import matplotlib.pyplot as plt

from pnr.vis.Event import Event, EventException
from pnr.data.utils import get_hoop_location

np.seterr(divide='raise', invalid='raise')


def convert_movement(movement):
    """
    Convert full court movement to half court movement
    """
    # movement.loc[movement.x_loc > 47, 'y_loc'] = movement.loc[movement.x_loc > 47, 'y_loc'].apply(lambda y: 50 - y)
    # movement.loc[movement.x_loc > 47, 'x_loc'] = movement.loc[movement.x_loc > 47, 'x_loc'].apply(lambda x: 94 - x)
    # movement['x_loc_copy'] = movement['x_loc']
    # movement['y_loc_copy'] = movement['y_loc']
    # movement['x_loc'] = movement['y_loc'].apply(lambda y: 250 * (1 - (y - 0) / (50 - 0)) + -250 * ((y - 0) / (50 - 0)))
    # movement['y_loc'] = movement['x_loc'].apply(lambda x: -47.5 * (1 - (x - 0) / (47 - 0)) + 422.5 * ((x - 0) / (47 - 0)))
    # movement = movement.drop('x_loc_copy', axis=1, inplace=False)
    # movement = movement.drop('y_loc_copy', axis=1, inplace=False)

    return movement


def convert_to_half(annotation_movements):
    for anno_ind, annotation in enumerate(annotation_movements):
        for player_ind, player in enumerate(annotation['players']):
            for role, movement in player['movement']['before'].items():
                movement = convert_movement(movement)
                player['movement']['before'][role] = copy(movement)
            for role, movement in player['movement']['after'].items():
                movement = convert_movement(movement)
                player['movement']['after'][role] = copy(movement)
            annotation_movements[anno_ind]['players'][player_ind] = player
    return annotation_movements


def create_trajectory(movement_dict, type):
    trajectory = {}
    trajectory['players'] = {}

    if type == 'before':
        screen_loc = movement_dict['screen_setter'][['x_loc', 'y_loc']].values[-1]
    elif type == 'after':
        screen_loc = movement_dict['screen_setter'][['x_loc', 'y_loc']].values[0]
    trajectory['screen_loc'] = screen_loc
    # create new increasing times for game_clock
    trajectory['game_clock'] = np.arange(0.0, 2.0, 0.04)

    for role, movement in movement_dict.items():
        trajectory['players'][role] = movement[['x_loc', 'y_loc']].values

    return trajectory

def extract_trajectories(annotation_movements):
    trajectories = []
    annotations = []

    for annotation in annotation_movements:
        for player in annotation['players']:


            # TODO edit and create dict for trajectory
            before_trajectory = create_trajectory(player['movement']['before'], 'before')
            after_trajectory = create_trajectory(player['movement']['after'], 'after')

            # append trajectory and annotation
            trajectories.append(before_trajectory)
            trajectories.append(after_trajectory)

            # append annotations for before and after for player
            annotation['annotation']['player_id'] = copy(player['player_id'])
            annotation['annotation']['action'] = 'before'
            before_annotation = copy(annotation['annotation'])
            annotations.append(before_annotation)
            annotation['annotation']['action'] = 'after'
            after_annotation = copy(annotation['annotation'])
            annotations.append(after_annotation)


    return trajectories, annotations


def euclid_dist(location_1, location_2):
    """
    Get L2 distance to ball from shot location
    """
    distance = (location_1-location_2)**2
    distance = distance.sum(axis=-1)
    distance = np.sqrt(distance)
    return distance


def over_under(player, screen_loc, hoop):
    """
    Get a +/-/0 score if the player is over/under/neither the screen.
    Players that are not close enough to the screen location, or are not
    under the perview of the radial range, are neither over/under the screen.
    Once a +1/-1/0 score is assigned, a range is applied to the player based
    on how close the player is.

    Parameters
    ----------
    player: np.array
        2d array with x,y positions
    screen_loc: 2d array
    hoop: 2d array

    Returns
    -------
    score: np.array
        over/under score
    """
    scores = []
    for loc in player:
        ba = loc - screen_loc
        bc = hoop - loc
        try:
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            rad = np.arccos(cosine_angle)
        except Exception:
            scores.append(0)
            continue

        if rad < np.pi / 2:
            screen_loc_dist = euclid_dist(screen_loc, hoop)
            player_loc_dist = euclid_dist(loc, hoop)

            dist = screen_loc_dist - player_loc_dist
            if dist < 10:
                if dist > 0:
                    # under
                    score = -1
                elif dist < 0:
                    # over
                    score = 1
                score = np.pi / 2 - rad
                score = (10 - dist) * score
            else:
                score = 0
        else:
            score = 0

        scores.append(score)

    return np.array(scores)

def complete_trajectories(trajectories, annotations):
    completed_trajectories = []
    completed_annotations = []

    for ind, trajectory in enumerate(trajectories):
        annotation = annotations[ind]
        completed_trajectory = []

        game_clock = trajectory['game_clock']
        player = trajectory['players']['player']
        ball_handler = trajectory['players']['ball_handler']
        ball_defender = trajectory['players']['ball_defender']
        screen_setter = trajectory['players']['screen_setter']
        screen_defender = trajectory['players']['screen_defender']
        screen_loc = np.full(player.shape, trajectory['screen_loc'])
        hoop = get_hoop_location(screen_loc[0][0])
        hoop = np.full(player.shape, hoop)

        distance_bh = euclid_dist(player, ball_handler)
        distance_bd = euclid_dist(player, ball_defender)
        distance_ss = euclid_dist(player, screen_setter)
        distance_sd = euclid_dist(player, screen_defender)
        distance_hoop = euclid_dist(player, hoop)
        distance_screen_loc = euclid_dist(player, screen_loc)
        over_under_score = over_under(player, screen_loc[0], hoop[0])

        for i in range(0, len(trajectory['players']['player'])):
            rec = []
            if i == 0:
                # time, location_c, speed_c,
                # distance_bh, distance_ss, over_under_score
                # rot_c
                rec = [game_clock[i], 0, 0, distance_bh[i], distance_ss[i], over_under_score[i], 0]
            else:
                loc_c = math.sqrt((player[i, 0] - player[i-1, 0])**2+(player[i, 1] - player[i-1, 1])**2)
                rec.append(game_clock[i])
                # rec.append(loc_c)
                # rec.append(loc_c / (game_clock[i] - game_clock[i - 1]))
                rec.append(distance_bh[i])
                rec.append(distance_ss[i])
                rec.append(over_under_score[i])
                # rec.append(distance_hoop[i])
                # rec.append(distance_screen_loc[i])
                # catch numpy exceptions
                try:
                    rec.append(math.atan((player[i, 1] - player[i-1, 1]) / (player[i, 0] - player[i-1, 0])))
                except Exception as err:
                    rec.append(0)
            completed_trajectory.append(rec)
        completed_trajectory = np.array(completed_trajectory)
        completed_trajectories.append(completed_trajectory)
        completed_annotations.append(annotation)

    return completed_trajectories, completed_annotations


def generate_behavior_sequences(features_trajectories):
    behavior_sequences = []

    for trajectory_features in tqdm(features_trajectories):
        # shape = 50, 8
        windows = rolling_window(trajectory_features)
        # shape = 5, 10, 8
        behavior_sequence = behavior_extract(windows)
        # shape = 5, 42
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
    for trajectory in completed_trajectories:
        trajectory_features = []
        for i in range(0,len(trajectory)):
            rec = []
            if i == 0:
                # time, loc_c_rate, diff_loc_c,
                # diff_dist_bh, diff_dist_ss,
                # diff_dist_hoop, diff_dist_screen_loc,
                # diff_rot_c
                rec = [0, 0, 0, 0, 0, 0]
            else:
                loc_c = trajectory[i][1]
                loc_c_rate = loc_c / (trajectory[i][0] - trajectory[i-1][0])
                rec.append(trajectory[i][0])
                # rec.append(loc_c_rate)
                rec.append(trajectory[i][1] - trajectory[i-1][1])
                rec.append(trajectory[i][2] - trajectory[i-1][2])
                rec.append(trajectory[i][3] - trajectory[i-1][3])
                rec.append(trajectory[i][4] - trajectory[i-1][4])
                # rec.append(trajectory[i][5] - trajectory[i - 1][5])
                # rec.append(trajectory[i][6] - trajectory[i - 1][6])
            trajectory_features.append(rec)
        trajectory_features = np.array(trajectory_features)
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
                behaviour_feature.append(description[4][i])
                # behaviour_feature.append(description[5][i])
                # behaviour_feature.append(description[6][i])

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
    action_movements = convert_to_half(action_movements)
    trajectories, annotations = extract_trajectories(action_movements)
    trajectories, annotations = complete_trajectories(trajectories, annotations)
    features = compute_features(trajectories)
    behaviours = generate_behavior_sequences(features)
    # behaviours = generate_normal_behavior_sequence(behaviours)

    return np.array(behaviours), annotations


def plot_pairwise_dist(action_movements):
    """
    Plot graph of different pairwise distances used in features

    Parameters
    ----------
    action_movements: dict
        before and after action movement information for annotation

    Returns
    -------
    """
    action_movements = convert_to_half(action_movements)
    trajectories, annotations = extract_trajectories(action_movements)

    for ind, trajectory in enumerate(trajectories):
        annotation = annotations[ind]
        game_id = int(annotation['gameid'])
        anno_id = int(annotation['eid'])
        anno = copy(annotation)

        game_pkl = os.path.join(game_dir, '00%s.pkl' % game_id)
        with open(game_pkl, 'rb') as f:
            raw_data = pd.read_pickle(f)

        e = Event(raw_data['events'][anno_id], anno=anno)
        ## render
        try:
            e.sequence_around_t(anno, 50, pnr=True)
            before = copy(e)
            after = copy(e)
            before.moments = before.moments[:50]
            after.moments = after.moments[50:]
            before.show_static(os.path.join(pnr_dir, 'pnr-before.pdf'), anno=anno)
            after.show_static(os.path.join(pnr_dir, 'pnr-after.pdf'), anno=anno)
        except EventException as e:
            print ('malformed sequence, skipping')
            continue

        game_clock = trajectory['game_clock']
        player = trajectory['players']['player']
        ball_handler = trajectory['players']['ball_handler']
        screen_setter = trajectory['players']['screen_setter']
        screen_loc = np.full(player.shape, trajectory['screen_loc'])
        hoop = get_hoop_location(screen_loc[0][0])
        hoop = np.full(player.shape, hoop)

        icons = ['r--', 'gs', 'b^', 'yv']
        distance_bh = euclid_dist(player, ball_handler)
        distance_ss = euclid_dist(player, screen_setter)
        distance_hoop = euclid_dist(player, hoop)
        distance_screen_loc = euclid_dist(player, screen_loc)

        fig, ax = plt.subplots()
        ax.plot(game_clock, distance_bh, icons[0], label='Ball Handler Distance')
        ax.plot(game_clock, distance_ss, icons[1], label='Screen Setter Distance')
        ax.plot(game_clock, distance_hoop, icons[2], label='Hoop Distance')
        ax.plot(game_clock, distance_screen_loc, icons[3], label='Screen Location Distance')

        ax.set_xlabel('Game Clock')
        ax.set_ylabel('Euclidean Distance')

        # Now add the legend with some customizations.
        if annotation['action'] == 'before':
            legend = ax.legend(loc='upper right', shadow=True)
        elif annotation['action'] == 'after':
            legend = ax.legend(loc='upper left', shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

        fig.show()
        if annotation['action'] == 'before':
            fig.savefig(os.path.join(pnr_dir, 'pairwise-distances-before.pdf'), format='pdf', bbox_inches='tight')
        elif annotation['action'] == 'after':
            fig.savefig(os.path.join(pnr_dir, 'pairwise-distances-after.pdf'), format='pdf', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    from pnr.data.constant import game_dir

    pnr_dir = os.path.join(game_dir, 'pnr-annotations')
    action_movements = pkl.load(open(os.path.join(pnr_dir, 'roles/trajectories.pkl'), 'rb'))
    # plot_pairwise_dist(action_movements)
    behaviours, annotations = get_behaviours(action_movements)

    np.save('%s/roles/behaviours' % (pnr_dir), behaviours)
    pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/annotations.pkl'), 'wb'))

