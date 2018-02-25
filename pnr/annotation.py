"""annotation.py

Usage:
    annotation.py <f_data_config>

Arguments:
    <f_data_config>  example 'pnrs.yaml'

Example:
    python annotation.py pnrs.yaml
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import math
import os
import cPickle as pickle
from docopt import docopt
import yaml

import pnr.config as CONFIG


def gameclock_to_str(gameclock):
    """
    Float to minute:second
    """
    return '%d:%d' % (int(gameclock/60), int(gameclock % 60))


def read_annotation(fpath):
    df = pd.read_csv(open(fpath,'rb'), header=None)
    annotations = {}
    for ind, row in df.iterrows():
        anno = []
        for pnr in row[1:]:
            if type(pnr) == str and ':' in pnr:
                m, s = pnr.split(':')
                anno.append(int(m) *60 + int(s))
            elif type(pnr) == str and pnr[0]=='-':
                pass # this is accepted format
            elif type(pnr) == float and math.isnan(pnr):
                pass # this is accepted format
            else:
                print pnr, pnr == '-'
                print('Error in game file: %s on line %i' % (fpath, ind))
                raise Exception('unknown annotation format')
        try:
            annotations[int(row[0])] = anno
        except Exception:
            continue
    return annotations


def read_annotation_from_raw(fpath, game_id):
    data = pickle.load(open(fpath, 'rb'))
    annotations = {}
    data = pd.DataFrame(data)
    event_ids = data.loc[:,'eid'].drop_duplicates(inplace=False).values

    for event in event_ids:
        eid_annotations = data.loc[data.eid == event, 'gameclock'].values
        annotations[event] = eid_annotations

    return annotations


def prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir):
    gt = []
    all_raw_f = filter(lambda s:'raw-' in s,os.listdir(pnr_dir))
    for pnr_anno_ind in xrange(len(all_raw_f)):
        game_anno_base = all_raw_f[pnr_anno_ind]
        if not os.path.isfile(os.path.join(pnr_dir,game_anno_base)):
            continue
        game_id = game_anno_base.split('.')[0].split('-')[1]
        raw_data = pd.read_pickle(os.path.join(game_dir, game_id+'.pkl'))
        fpath = os.path.join(pnr_dir, game_anno_base)
        anno = read_annotation(fpath)
        for k, v in anno.items():
            if len(v) == 0:
                continue
            gt_entries = []
            q = raw_data['events'][k]['quarter']
            for vi in v:
                gt_entries.append({'gameid':game_id, 'quarter':q, 'gameclock':vi, 'eid':k})
            gt += gt_entries
    return gt


def script_anno_rev0():
    gt = prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir)
    pickle.dump(gt, open(os.path.join(pnr_dir,'gt/rev0.pkl'),'wb'))
    annotations = pd.DataFrame(gt)
    annotations.to_csv(os.path.join(pnr_dir, 'gt/annotations.csv'), index=False)


def raw_to_gt_format():
    """
    Read raw annotation files to convert to output similar to the output from make_raw_from_untrained
    Use for future comparison purposes
    """
    if not os.path.exists('%s/raw/' % pnr_dir):
        os.makedirs('%s/raw/' % pnr_dir)

    annotations = pd.read_csv(os.path.join(pnr_dir, 'gt/annotations.csv'))
    game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values
    for game_id in game_ids:
        game_annotations = annotations.loc[annotations.gameid == game_id, ['eid', 'gameclock', 'gameclock_str', 'quarter']]
        game_annotations.to_csv('%s/raw/raw-00%s.csv' % (pnr_dir, game_id),index=False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    from pnr.data.constant import data_dir, game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')
    script_anno_rev0()
    raw_to_gt_format()
