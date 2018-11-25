import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import os

from pnr.data.utils import limit_to_half
import pnr.config as CONFIG

movement_headers = [
    "team_id",
    "player_id",
    "x_loc",
    "y_loc",
    "radius",
    "game_clock",
    "shot_clock",
    "quarter",
    "game_id",
    "event_id"
]

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
            (movement.game_clock <= (annotation['gameclock'] + 0.6 + int(int(data_config['data_config']['tfr']) / int(data_config['data_config']['frame_rate'])))) &
            (movement.game_clock >= (annotation['gameclock'] + 0.6)) &
            (movement.quarter == annotation['quarter'])
        , :]
    elif annotation['action'] == 'after':
        movement = movement.loc[
            (movement.game_clock >= (annotation['gameclock'] + 0.6 - int(data_config['data_config']['tfr'] / int(data_config['data_config']['frame_rate'])))) &
            (movement.game_clock <= (annotation['gameclock'] + 0.6)) &
            (movement.quarter == annotation['quarter'])
        , :]

    screen_loc = movement.loc[movement.player_id == annotation['screen_setter'], ['x_loc', 'y_loc']].values[-1]
    movement = movement.loc[movement.player_id == annotation['player_id'], :]
    movement = limit_to_half(movement, screen_loc)

    movement = full_to_half_full(movement)
    movement = half_full_to_half(movement)

    fig = plt.figure(figsize=(12, 11))
    plt.scatter(movement.x_loc, movement.y_loc, c=movement.game_clock, cmap=plt.cm.Blues, s=250, zorder=1)

    draw_half_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250, 250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)
    # get rid of axis tick labels
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.show()

    dir = '%s/%s/plots/%s/' % (CONFIG.plots.dir, 'actions', annotation['label'])
    if not os.path.exists(dir):
        os.makedirs(dir)

    fig.savefig('%s/%s/plots/%s/00%s_%s_%s_%s.png' % (
        CONFIG.plots.dir,
        'actions',
        str(int(annotation['label'])),
        str(int(annotation['gameid'])),
        str(int(annotation['eid'])),
        str(int(annotation['gameclock'])),
        str(int(annotation['player_id']))
    ))
    plt.close()


def draw_half_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def half_full_to_half(data):

    # convert to half court scale
    # note the x_loc and the y_loc are switched in shot charts from movement data (charts are perpendicular)
    data['x_loc_copy'] = data['x_loc']
    data['y_loc_copy'] = data['y_loc']

    # Range conversion formula
    # http://math.stackexchange.com/questions/43698/range-scaling-problem

    data['x_loc'] = data['y_loc_copy'].apply(lambda y: 250 * (1 - (y - 0)/(50 - 0)) + -250 * ((y - 0)/(50 - 0)))
    data['y_loc'] = data['x_loc_copy'].apply(lambda x: -47.5 * (1 - (x - 0)/(47 - 0)) + 422.5 * ((x - 0)/(47 - 0)))
    data = data.drop('x_loc_copy', axis=1, inplace=False)
    data = data.drop('y_loc_copy', axis=1, inplace=False)

    return data


def full_to_half_full(data):

    # first force all points above 47 to their half court counterparts
    data.loc[data.x_loc > 47,'y_loc'] = data.loc[data.x_loc > 47, 'y_loc'].apply(lambda y: 50 - y)
    data.loc[data.x_loc > 47,'x_loc'] = data.loc[data.x_loc > 47, 'x_loc'].apply(lambda x: 94 - x)

    return data