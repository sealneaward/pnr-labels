import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_possession_team(player, movement):
    """
    Return the team_id of the ball_handler
    """
    team_id = movement.loc[movement.player_id == player, 'team_id'].values[0]
    return team_id


def get_ball_handler(movement):
    """
    Use ball location to MSE of player location

    Parameters
    ----------
    ball_location: np.array
        x/y location data
    player_location: np.array
        x/y location data

    Returns
    -------
    distance: np.array
        difference in locations to use to find ball handler
    """
    movement['distance_to_ball'] = 0
    ball_movement = movement.loc[movement.player_id == -1, :]
    players_movement = movement.loc[movement.player_id != -1, :]
    smallest_distance = 999999
    ball_handler = None

    for player_id, player_movement in players_movement.groupby('player_id'):

        player_movement['shot_location_x'] = ball_movement['x_loc'].values
        player_movement['shot_location_y'] = ball_movement['y_loc'].values

        mse = mean_squared_error(
            (
                player_movement[['x_loc', 'y_loc']].values
            ),
            (
                player_movement[['shot_location_x', 'shot_location_y']].values
            )
        )

        if smallest_distance > mse:
            smallest_distance = mse
            ball_handler = player_id

    return ball_handler


def get_screen_setter(ball_handler, ball_handler_team, movement, annotation):
    """
    Use radius from ball to find offensive player not ball handler
    that is the player setting the screen

    Parameters
    ----------
    ball_handler: int
        player id
    ball_handler_team: int
        team id
    movement: pd.DataFrame
        sportvu movement data
    annotation: dict
        pnr information

    Returns
    -------
    screen_setter: int
        player id
    """
    # get closest time to screen annotation time
    game_clocks = movement['game_clock'].drop_duplicates(inplace=False).values
    game_clock = game_clocks[np.argmin(np.abs(game_clocks - (annotation['gameclock'] + 0.6)))]
    screen_setter = None

    movement = movement.loc[movement.game_clock == game_clock, :]
    ball_handler_movement = movement.loc[movement.player_id == ball_handler, :]
    players_movement = movement.loc[
       (movement.player_id != ball_handler) &
       (movement.player_id != -1) &
       (movement.team_id == ball_handler_team)
    , :]

    smallest_distance = 999999
    for player_id, player_movement in players_movement.groupby('player_id'):

        player_movement['ball_handler_location_x'] = ball_handler_movement['x_loc'].values
        player_movement['ball_handler_location_y'] = ball_handler_movement['y_loc'].values

        mse = mean_squared_error(
            (
                player_movement[['x_loc', 'y_loc']].values
            ),
            (
                player_movement[['ball_handler_location_x', 'ball_handler_location_y']].values
            )
        )

        if smallest_distance > mse:
            smallest_distance = mse
            screen_setter = player_id

    return screen_setter


def get_ball_defender(ball_handler, ball_handler_team, movement):
    """
    Use ball location to MSE of player location

    Parameters
    ----------
    ball_location: np.array
        x/y location data
    player_location: np.array
        x/y location data

    Returns
    -------
    distance: np.array
        difference in locations to use to find ball handler
    """
    movement['distance_to_ball'] = 0
    ball_handler_movement = movement.loc[movement.player_id == ball_handler, :]
    players_movement = movement.loc[
        (movement.player_id != ball_handler) &
        (movement.player_id != -1) &
        (movement.team_id != ball_handler_team)
    , :]
    ball_defender = None

    smallest_distance = 999999
    for player_id, player_movement in players_movement.groupby('player_id'):

        player_movement['ball_handler_location_x'] = ball_handler_movement['x_loc'].values
        player_movement['ball_handler_location_y'] = ball_handler_movement['y_loc'].values

        mse = mean_squared_error(
            (
                player_movement[['x_loc', 'y_loc']].values
            ),
            (
                player_movement[['ball_handler_location_x', 'ball_handler_location_y']].values
            )
        )

        if smallest_distance > mse:
            smallest_distance = mse
            ball_defender = player_id

    return ball_defender


def get_screen_defender(screen_setter, ball_defender, screen_setter_team, movement):
    """
    Use ball location to MSE of player location

    Parameters
    ----------
    ball_location: np.array
        x/y location data
    player_location: np.array
        x/y location data

    Returns
    -------
    distance: np.array
        difference in locations to use to find ball handler
    """
    movement['distance_to_ball'] = 0
    screen_setter_movement = movement.loc[movement.player_id == screen_setter, :]
    players_movement = movement.loc[
        (movement.player_id != ball_defender) &
        (movement.player_id != -1) &
        (movement.team_id != screen_setter_team)
    , :]
    screen_defender = None

    smallest_distance = 999999
    for player_id, player_movement in players_movement.groupby('player_id'):

        player_movement['ball_handler_location_x'] = screen_setter_movement['x_loc'].values
        player_movement['ball_handler_location_y'] = screen_setter_movement['y_loc'].values

        mse = mean_squared_error(
            (
                player_movement[['x_loc', 'y_loc']].values
            ),
            (
                player_movement[['ball_handler_location_x', 'ball_handler_location_y']].values
            )
        )

        if smallest_distance > mse:
            smallest_distance = mse
            screen_defender = player_id

    return screen_defender


def get_roles(annotation, movement, data_config):
    """
    Get 4 roles from movement to satisfy 4 roles in PnR

    Parameters
    ----------
    annotation: pd.DataFrame
        pnr information
    movement: pd.DataFrame
        sportvu movement information
    data_config: dict
        configuration information
    """
    annotation_movement = movement.loc[
        (movement.game_clock <= (annotation['gameclock'] + 0.6)) &
        (movement.game_clock >= (annotation['gameclock'] + 0.6 - int(data_config['data_config']['tfr']/data_config['data_config']['frame_rate'])))
    , :]

    ball_handler_id = get_ball_handler(annotation_movement)
    if ball_handler_id is None:
        return None
    ball_handler_team = get_possession_team(ball_handler_id, annotation_movement)

    ball_defender_id = get_ball_defender(ball_handler_id, ball_handler_team, annotation_movement)
    if ball_defender_id is None:
        return None
    ball_defender_team = get_possession_team(ball_defender_id, annotation_movement)

    screen_setter_id = get_screen_setter(ball_handler_id, ball_handler_team, annotation_movement, annotation)
    if screen_setter_id is None:
        return None
    screen_setter_team = get_possession_team(screen_setter_id, annotation_movement)

    screen_defender_id = get_screen_defender(screen_setter_id, ball_defender_id, screen_setter_team, annotation_movement)
    if screen_defender_id is None:
        return None
    screen_defender_team = get_possession_team(screen_defender_id, annotation_movement)

    annotation['ball_handler'] = ball_handler_id
    annotation['ball_defender'] = ball_defender_id
    annotation['screen_setter'] = screen_setter_id
    annotation['screen_defender'] = screen_defender_id
    annotation['offense_id'] = ball_handler_team
    annotation['defense_id'] = ball_defender_team

    return annotation