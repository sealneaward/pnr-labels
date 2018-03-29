import pandas as pd
import os
import cPickle as pkl

from pnr.annotation.behaviour import *

def get_action_movement(movement, annotation, data_config):
    """
    Get movement of actions before and after the screen time.
    There essentially will be two actions for each player in the annotation.

    Parameters
    ----------
    movement: pandas.DataFrame
        sportvu movement data
    data_config: dict
        annotation configuration
    annotation: dict
        annotation information

    Returns
    -------
    action_movements: dict
        before and after action movement information
    """
    action_movements = {}
    action_movements['annotation'] = annotation
    action_movements['players'] = []

    player_ids = [
        annotation['ball_handler'],
        annotation['ball_defender'],
        annotation['screen_setter'],
        annotation['screen_defender']
    ]
    for player_id in player_ids:
        player_movement = movement.loc[movement.player_id == player_id, :]
        before_movement = player_movement.loc[
            (player_movement.game_clock <= annotation['gameclock'] + 0.6 + int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate'])) &
            (player_movement.game_clock >= annotation['gameclock'] + 0.6)
        , :]
        after_movement = player_movement.loc[
            (player_movement.game_clock >= annotation['gameclock'] + 0.6 - int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate'])) &
            (player_movement.game_clock <= annotation['gameclock'] + 0.6)
        , :]

        before_movement = before_movement[:data_config['data_config']['tfr']]
        after_movement = after_movement[:data_config['data_config']['tfr']]

        if len(before_movement) != data_config['data_config']['tfr'] or len(after_movement) != data_config['data_config']['tfr']:
            return None

        player_movement = {'before': before_movement, 'after': after_movement}
        action_movements['players'].append({'player_id': player_id, 'movement': player_movement})

    return action_movements


def get_actions(annotation, movement, data_config):
    """
    Get actions split by the screen time from movement using unsupervised methods

    Parameters
    ----------
    annotation: pd.DataFrame
        pnr information
    movement: pd.DataFrame
        sportvu movement information
    data_config: dict
        configuration information

    Returns
    -------
    action_movements: dict
        before and after action movement information for annotation
    """
    from pnr.data.constant import game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    annotation_movement = movement.loc[
      (movement.game_clock <= (annotation['gameclock'] + 0.6 + int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate']))) &
      (movement.game_clock >= (annotation['gameclock'] + 0.6 - int(data_config['data_config']['tfr'] / data_config['data_config']['frame_rate'])))
    , :]

    action_movements = get_action_movement(annotation_movement, annotation, data_config)
    return action_movements