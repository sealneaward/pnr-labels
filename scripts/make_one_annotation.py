"""make_one_annotation.py

Usage:
    make_one_annotation.py <game_id> <anno_id> <dir-prefix> <pnr-prefix> <time-frame-radius> <raw_file>

Arguments:
    <dir-prefix> the prefix prepended the directory that will be created to hold the videos
    <pnr-prefix> the prefix for annotation filenames (e.g. 'raw')
    <time-frame-radius> tfr, let annotated event be T_a, we extract frames [T_a-tfr, T_a+tfr]
    <game_id> game file
    <anno_id> annotation
    <raw_file> location of annotation file

Example:
    python make_one_annotation.py 0021500383 3 viz raw 50 rev0.pkl
"""

from pnr.annotation import annotation
from pnr import data
from pnr.vis.Event import Event, EventException

from copy import copy
import os
from docopt import docopt
import pandas as pd


def wrapper_render_one_anno(dir_prefix, gameid, anno_id):
    ### Load game
    print ('Loading')
    game_basename = gameid+'.pkl'

    game_pkl = os.path.join(game_dir, game_basename)
    with open(game_pkl,'rb') as f:
        raw_data = pd.read_pickle(f)
    game_str = "{visitor}@{home}, on {date}".format(
        visitor=raw_data['events'][0]['visitor']['abbreviation'],
        home=raw_data['events'][0]['home']['abbreviation'],
        date=raw_data['gamedate']
    )
    print (game_str)


    ### Create a new directory for videos
    vid_dir =os.path.join(game_dir, 'video') # base dir that holds all the videos
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    new_dir = os.path.join(vid_dir, '{prefix}-{game_id}'.format(
        prefix=dir_prefix,
        game_id=game_basename.split('.')[0]
    ))
    previous_rendered_events = []
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else: # already a directory exists, likely we've tried to do the same thing
        print(new_dir)
        print('Already exists, not rerunning events rendered and saved previously')

    render_one_anno(
        raw_data,
        new_dir,
        anno_id
    )


def render_one_anno(raw_data, directory, anno_id):
    """
    Input:
        raw_data: the huge dictionary of a single game
    """
    N = len(raw_data['events'])
    anno_id = int(anno_id)
    pnr_annotations = annotation.read_annotation_from_raw(os.path.join(pnr_dir, 'roles/%s' % (arguments['<raw_file>'])), raw_data['gameid'])
    annos = pnr_annotations[anno_id]

    for ind, anno in enumerate(annos):
        e = Event(raw_data['events'][anno_id], anno=anno)
        ## render
        try:
            e.sequence_around_t(anno, int(arguments['<time-frame-radius>']), pnr=True)
            before = copy(e)
            after = copy(e)
            before.moments = before.moments[:int(arguments['<time-frame-radius>'])]
            after.moments = after.moments[int(arguments['<time-frame-radius>']):]
            before.show_static(os.path.join(directory, '%i-pnr-%i-before.pdf' %(anno_id, ind)), anno=anno)
            after.show_static(os.path.join(directory, '%i-pnr-%i-after.pdf' % (anno_id, ind)), anno=anno)
        except EventException as e:
            print ('malformed sequence, skipping')
            continue


if __name__ == '__main__':
    game_dir = data.constant.game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__, version='something 1.1.1')
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    game_id = arguments['<game_id>']
    anno_id = arguments['<anno_id>']
    dir_prefix = arguments['<dir-prefix>']
    wrapper_render_one_anno(dir_prefix, game_id, anno_id)