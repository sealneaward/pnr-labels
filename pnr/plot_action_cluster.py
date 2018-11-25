"""plot_action_cluster.py

Usage:
    plot_action_cluster.py <f_data_config> <action_file>

Arguments:
    <f_data_config>  example ''pnrs.yaml''
    <action_file>  example ''distances_up_down_score_5.csv''

Example:
    python plot_action_cluster.py pnrs.yaml distances_up_down_score_5.csv
"""
from docopt import docopt
import os
import pandas as pd
import json
import numpy as np
import yaml
import matplotlib.pyplot as plt

import pnr.config as CONFIG
from pnr.plots.plot import full_to_half_full, half_full_to_half, limit_to_half, draw_half_court, movement_headers

def plot_action(game, annotation, game_id, data_config):
    """
    """

    moments = []
    movement_data = game['events'][int(annotation['eid'])]['moments']
    for moment in movement_data:
        for player in moment[5]:
            player.extend([moment[2], moment[3], moment[0], game_id, annotation['eid']])
            player = player[:len(movement_headers)]
            moments.append(player)

    movement = pd.DataFrame(data=moments, columns=movement_headers)

    if annotation['action'] == 'before':
        movement = movement.loc[
            (movement.game_clock <= (annotation['gameclock'] + 0.6 + int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate']))) &
            (movement.game_clock >= annotation['gameclock'] + 0.6) &
            (movement.quarter == annotation['quarter'])
        , :]
    elif annotation['action'] == 'after':
        movement = movement.loc[
            (movement.game_clock >= (annotation['gameclock'] + 0.6 - int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate']))) &
            (movement.game_clock <= annotation['gameclock'] + 0.6) &
            (movement.quarter == annotation['quarter'])
        , :]

    screen_loc = movement.loc[movement.player_id == annotation['screen_setter'], ['x_loc', 'y_loc']].values[-1]
    movement = movement.loc[movement.player_id == annotation['player_id'], :]
    movement = limit_to_half(movement, screen_loc)

    movement = full_to_half_full(movement)
    movement = half_full_to_half(movement)


    plt.scatter(movement.x_loc, movement.y_loc, c=movement.game_clock, cmap=plt.cm.Blues, s=100, zorder=1)


def plot_actions(data_config, actions, action_id):
    """
    Read csv with actions to plot in single representation as "mean" of sorts

    Parameters
    ----------
    data_config: dict
        data configuration information
    action_file: pandas.DataFrame
        action information: game, eid, etc
    annotations: pandas.DataFrame
        raw annotation information
    """
    fig = plt.figure(figsize=(12, 11))
    plt.tick_params(labelbottom=False, labelleft=False)
    game_ids = actions['gameid'].drop_duplicates(inplace=False).values

    for game_id in game_ids:
        game_id = '00' + str(int(game_id))
        game = pd.read_pickle(os.path.join(game_dir, game_id + '.pkl'))
        game_id = int(game_id)
        game_action_annotations = actions.loc[actions.gameid == game_id, :]
        for ind, action in game_action_annotations.iterrows():
            plot_action(game, action, game_id, data_config)

    draw_half_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250, 250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)

    fig.show()
    fig.savefig('%s/%s.pdf' % (pnr_dir, action_id), format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    from pnr.data.constant import sportvu_dir, game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    action_file = '%s/%s' % (pnr_dir, arguments['<action_file>'])
    actions = pd.read_csv(action_file)
    action_id = int(arguments['<action_file>'].split('_')[-1].split('.')[0])

    annotations = pd.read_csv('%s/roles/clusters.csv' % (pnr_dir))
    actions = pd.merge(
        left=actions,
        right=annotations,
        on=['gameid', 'eid', 'gameclock', 'player_id']
    )
    actions = actions.loc[actions.label == action_id]

    plot_actions(data_config, actions, action_id)