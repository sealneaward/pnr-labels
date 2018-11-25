"""make_one_game.py

Usage:
    make_one_game.py <f_data_config> <gameid> <index> <dir-prefix> <pnr-prefix> <time-frame-radius> <raw_file>

Arguments:
    <index> not a very good way of doing things, this is the index into os.listdir
    <dir-prefix> the prefix prepended the directory that will be created to hold the videos
    <pnr-prefix> the prefix for annotation filenames (e.g. 'raw')
    <time-frame-radius> tfr, let annotated event be T_a, we extract frames [T_a-tfr, T_a+tfr]
    <raw_file> location of annotation file

Example:
    python make_one_game.py pnrs.yaml 0021500408 1 topics raw 75 from-raw-examples.pkl
"""

from pnr.annotation import annotation
from pnr import data
from pnr.vis.Event import Event, EventException
import pnr.config as CONFIG

import os
import yaml
from docopt import docopt
import pandas as pd
import numpy as np


def wrapper_render_one_game(index, dir_prefix, gameid=None):
    ### Load game
    print ('Loading')
    if gameid != None:
        game_basename = gameid+'.pkl'
    else:
        game_basename = os.listdir(game_dir)[index]

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
    vid_dir = os.path.join(game_dir, 'video') # base dir that holds all the videos
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    new_dir = os.path.join(vid_dir, '{prefix}-{game_id}'.format(
        prefix=dir_prefix,
        game_id=game_basename.split('.')[0]
    ))
    new_dir = '%s/%s_%s' % (new_dir, data_config['embedded_type'], data_config['n_clusters'])

    previous_rendered_events = []
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else: # already a directory exists, likely we've tried to do the same thing
        print(new_dir)
        print('Already exists, not rerunning events rendered and saved previously')
        previous_rendered_events = os.listdir(new_dir)

    render_one_game(
        raw_data,
        new_dir,
        [int(name.split('.')[0].split('-')[0]) for name in previous_rendered_events]
    )


def render_one_game(raw_data, directory, skip_these):
    """
    Input:
        raw_data: the huge dictionary of a single game
    """
    N = len(raw_data['events'])
    pnr_annotations = annotation.read_annotation_from_raw(os.path.join(pnr_dir, 'roles/%s' % (arguments['<raw_file>'])), raw_data['gameid'])

    annotations = pd.read_csv('%s/%s/raw-%s.csv' % (pnr_dir, 'extended', arguments['<gameid>']))
    annotations[['over', 'under', 'switch', 'trap']] = annotations[['over', 'under', 'switch', 'trap']].fillna(0)
    annotations[['over', 'under', 'switch', 'trap']] = annotations[['over', 'under', 'switch', 'trap']].replace('x', 1)

    for i in xrange(N):
        if i in skip_these:
            print ('Skipping event <%i>' % i)
            continue
        else:
            if i not in pnr_annotations.keys():
                print "Clip index %i not labelled" % i
                continue
            else:
                annos = pnr_annotations[i]

        for ind, anno in enumerate(annos):
            ## preprocessing
            true_annotation = annotations.loc[
                (annotations.eid == anno['eid']) &
                (annotations.gameclock == anno['gameclock'])
            ,:]

            y_true = true_annotation[['over', 'under', 'switch', 'trap']].values
            y_true = np.argmax(y_true, axis=1)

            e = Event(raw_data['events'][i], anno=anno)
            ## render
            try:
                new_dir = '%s/%s' % (directory, e.anno['topic'])
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                e.sequence_around_t(anno, int(arguments['<time-frame-radius>']), pnr=True)
                e.show(os.path.join(new_dir, '%i-pnr-%i-%i.mp4' %(i, int(e.anno['gameclock']), int(y_true))), anno=anno)
            except Exception as e:
                print ('malformed sequence, skipping')
                continue


if __name__ == '__main__':
    game_dir = data.constant.game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__, version='something 1.1.1')
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))


    gameid = arguments['<gameid>']
    index = arguments['<index>']
    if index != None:
        index = int(index)
    dir_prefix = arguments['<dir-prefix>']
    wrapper_render_one_game(index, dir_prefix, gameid)