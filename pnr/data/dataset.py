from __future__ import division
import cPickle as pickle
import yaml
import os
from pnr.data.constant import data_dir, game_dir
import numpy as np
import pandas as pd


def _hash(i):
    return i['gameclock'] + i['eid']


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def disentangle_train_val(train, val):
    """
    Given annotations of train/val splits, make sure no overlapping Event
    -> for later detection testing
    """
    new_train = []
    new_val = []
    while len(val) > 0:
        ve = val.pop()
        vh = _hash(ve)
        if vh in [_hash(i) for i in train] + [_hash(i) for i in new_train]:
            new_train.append(ve)
            # to balance, find a unique train_anno to put in val
            while True:
                te = train.pop(0)
                if _hash(te) in [_hash(i) for i in train]:  # not unique, put back
                    train.append(te)
                else:
                    new_val.append(te)
                    break
        else:
            new_val.append(ve)
    new_train += train
    return new_train, new_val


class BaseDataset:
    """base class for loading the dataset
    """

    def __init__(self, f_config, fold_index, load_raw=True, no_anno=False, game=None):
        # configuration
        self.fold_index = fold_index
        if type(f_config) == str:
            self.config = yaml.load(open(f_config, 'rb'))
        else:
            self.config = f_config
        assert (fold_index >= 0 and fold_index <
                self.config['data_config']['N_folds'])
        self.tfr = self.config['data_config']['tfr']
        self.t_jitter = self.config['data_config']['t_jitter']
        self.t_negative = self.config['data_config']['t_negative']
        self.game_ids = self.config['data_config']['game_ids']
        self.n_classes = self.config['n_classes']
        ###
        if load_raw == True:
            self.games = {}
            for gameid in self.game_ids:
                raw_data = pd.read_pickle(os.path.join(game_dir, gameid + '.pkl'))
                self.games[raw_data['gameid']] = raw_data
        if not no_anno:
            self.annotations = pickle.load(open('%s/%s' % (game_dir,self.config['data_config']['annotation'])))
            # removing 0.6 second delay because of professionalism in raptors data
            # if only using one game, limit annotations for single game
            if game is not None:
                self.annotations = pd.DataFrame.from_records(self.annotations)
                self.annotations = self.annotations.loc[self.annotations.gameid == game, :]
                self.annotations = self.annotations.T.to_dict().values()

            self.train_annotations, self.val_annotations = self._split(self.annotations, fold_index)
            # make sure no overlapping Event between train and val
            self.train_annotations, self.val_annotations = disentangle_train_val(self.train_annotations,
                                                                                 self.val_annotations)
            self._make_annotation_dict()
            # hard negative examples
            ## WARNING: K-fold not supported here
            ##  user needs to make sure the following pkl
            ##  comes from the correct fold
            if 'hard-negatives' in self.config['data_config']:
                self.hard_negatives = pickle.load(open(data_dir + self.config['data_config']['hard-negatives']))
            if load_raw:
                # annotation only has Events with PNR
                # need to find Events without PNR, split into train/val
                #   for negatives examples
                # Let's call these Events 'void'
                # use gameclock = -1
                self.voids = []
                for gameid in self.game_ids:
                    curr_game_events = self.games[gameid]['events']
                    for eid, event in enumerate(curr_game_events):
                        if not eid in self.annotation_dict_eids[gameid][event['quarter']]:
                            self.voids.append(
                                self._make_annotation(
                                    gameid,
                                    event['quarter'],
                                    -1,
                                    eid,
                                    self.config['data_config']['tfr']
                                )
                            )
                self.train_voids, self.val_voids = self._split(self.voids, fold_index)

                # loader use below for detection task
                # hash validation events
                self.val_hash = {}
                for va in self.val_annotations + self.val_voids:
                    k = _hash(va)
                    if k not in self.val_hash:
                        self.val_hash[k] = []
                    self.val_hash[k].append(va)
                # hash train events for mining hard examples
                self.train_hash = {}
                for va in self.train_annotations + self.train_voids:
                    k = _hash(va)
                    if k not in self.train_hash:
                        self.train_hash[k] = []
                    self.train_hash[k].append(va)
                if game is not None:
                    self.game_hash = merge_two_dicts(self.train_hash, self.val_hash)

        self.val_ind = 0
        self.train_ind = 0
        self.ind = 0

    def _split(self, annotations, fold_index):
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(annotations)
        N = len(annotations)
        val_start = np.round(
            fold_index / self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1) /
                           self.config['data_config']['N_folds'] * N).astype('int32')
        val_annotations = annotations[val_start:val_end]
        train_annotations = annotations[:val_start] + annotations[val_end:]
        return train_annotations, val_annotations

    def _make_annotation_dict(self):
        self.annotation_dict = {}
        self.annotation_dict_eids = {}
        for anno in self.annotations:
            if anno['gameid'] not in self.annotation_dict:
                self.annotation_dict[anno['gameid']] = {}
                self.annotation_dict_eids[anno['gameid']] = {}
            if anno['quarter'] not in self.annotation_dict[anno['gameid']]:
                self.annotation_dict[anno['gameid']][anno['quarter']] = []
                self.annotation_dict_eids[anno['gameid']][anno['quarter']] = []
            self.annotation_dict[anno['gameid']][anno['quarter']].append(anno['gameclock'])
            self.annotation_dict_eids[anno['gameid']][anno['quarter']].append(anno['eid'])
        for game in self.annotation_dict.values():
            for quarter_ind in game.keys():
                game[quarter_ind] = np.sort(game[quarter_ind])[::-1]

    def propose_positive_Ta(self, jitter=True, train=True, loop=False):
        if not loop:  # sampling from training annotations
            while True:
                r_ind = np.random.randint(0, len(self.train_annotations))
                if not jitter:
                    ret = self.train_annotations[r_ind]
                else:
                    anno = self.train_annotations[r_ind].copy()
                    anno['gameclock'] += np.random.rand() * self.t_jitter
                    ret = anno

                # check not too close to boundary (i.e. not enough frames to
                # make a sequence)
                e = self.games[anno['gameid']]['events'][ret['eid']]
                for idx, moment in enumerate(e['moments']):
                    # seek
                    if moment[2] < ret['gameclock']:
                        if idx + self.tfr <= len(e['moments']) and idx - self.tfr >= 0:
                            return ret
                        else:
                            break  # try again...

        else:
            if train:
                if self.train_ind == len(self.train_annotations):  # end, reset
                    self.train_ind = 0
                    return None
                else:
                    ret = self.train_annotations[self.train_ind]
                    self.train_ind += 1
                    return ret
            else:
                if self.val_ind == len(self.val_annotations):  # end, reset
                    self.val_ind = 0
                    return None
                else:
                    ret = self.val_annotations[self.val_ind]
                    self.val_ind += 1
                    return ret

    def _make_annotation(self, gameid, quarter, gameclock, eid, tfr):
        return {
            'gameid': gameid,
            'quarter': int(quarter),
            'gameclock': float(gameclock),
            'start_time': float(gameclock + tfr / 25),
            'end_time': float(gameclock - tfr / 25),
            'eid': int(eid)
        }

    def propose_Ta(self):
        while True:
            g_ind = np.random.randint(0, len(self.games))
            e_ind = np.random.randint(0, len(self.games[self.games.keys()[g_ind]]['events']))

            e = self.games[self.games.keys()[g_ind]]['events'][e_ind]
            if len(e['moments']) < self.tfr * 2:
                continue
            try:
                m_ind = np.random.randint(self.tfr, len(e['moments']) - self.tfr)
            except Exception as err:
                m_ind = self.tfr

            return self._make_annotation(
                self.games.keys()[g_ind],
                e['quarter'],
                float(e['moments'][m_ind][2]),
                e_ind,
                self.config['data_config']['tfr']
            )

    def propose_negative_Ta(self, train=True):
        ## make sure it's in the right split
        if train:
            annos = self.train_annotations + self.train_voids
        else:
            annos = self.val_annotations + self.val_voids
        split_hash = [_hash(anno) for anno in annos]
        while True:
            cand = self.propose_Ta()
            if _hash(cand) not in split_hash:
                continue
            g_cand = int(cand['gameclock'])
            positives = self.annotation_dict[cand['gameid']][cand['quarter']]
            # too close to a positive label
            diff = np.min(np.abs(positives - g_cand))
            if diff < self.t_negative:
                # print ('proporal positive')
                continue
            return cand
