from __future__ import division
import cPickle as pickle
import yaml
import os
import numpy as np
import pandas as pd
import cPickle as pkl

from pnr.vis.Event import Event, EventException
from pnr.vis.Team import TeamNotFoundException
from pnr.data.extractor import ExtractorException, OneHotException
from pnr.data.utils import shuffle_2_array, make_3teams_11players
from pnr.data.constant import data_dir, game_dir
import pnr.config as CONFIG
from pnr.data.constant import game_dir


class BaseLoader:
    def __init__(self, data_config, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        self.data_config = data_config
        self.dataset = dataset
        self.extractor = extractor
        self.batch_size = batch_size
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.event_index = 0
        self.games = self.dataset.games

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True, no_anno=False):
        N_pos = int(self.fraction_positive * self.batch_size)
        N_neg = self.batch_size - N_pos
        ret_val = []
        if not no_anno:
            func = [self.dataset.propose_positive_Ta,
                    self.dataset.propose_negative_Ta]
        else:
            func = [self.dataset.propose_Ta]
        Ns = [N_pos, N_neg]
        # anno = func[0]()
        for j in xrange(len(func)):
            for _ in xrange(Ns[j]):
                while True:
                    try:
                        anno = func[j]()
                        e = Event(self.dataset.games[anno['gameid']][
                                  'events'][anno['eid']], gameid=anno['gameid'])
                        e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                        if extract:
                            # ExtractorException
                            ret_val.append(self.extractor.extract(e))
                        else:
                            # just to make sure event not malformed (like
                            # missing player)
                            _ = self.extractor.extract_raw(e)
                            ret_val.append(e)
                    except EventException as exc:
                        pass
                    except ExtractorException as exc:
                        pass
                    except TeamNotFoundException as exc:
                        pass
                    else:
                        break

        return (
                np.array(ret_val),
                np.vstack([np.array([[0, 1]]).repeat(N_pos, axis=0),
                np.array([[1, 0]]).repeat(N_neg, axis=0)])
            )

    def load_by_annotations(self, annotations, extract=True):
        """
        no labels returned
        """
        ret_val = []
        ret_labels = []
        self.extractor.augment = False
        for anno in annotations:
            try:
                e = Event(self.dataset.games[anno['gameid']][
                          'events'][anno['eid']], gameid=anno['gameid'])
                e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            except TeamNotFoundException as exc:
                pass
        return ret_val

    def load_split(self, split='val', extract=True, positive_only=False):
        N_pos = 0
        ret_val = []
        ret_labels = []
        self.extractor.augment = False
        istrain = split == 'train'
        while True:
            anno = self.dataset.propose_positive_Ta(jitter=False, train=istrain, loop=True)
            if anno == None:
                break
            try:
                e = Event(self.dataset.games[anno['gameid']]['events'][anno['eid']], anno=anno, gameid=anno['gameid'])
                e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except TeamNotFoundException as exc:
                pass
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            else:
                N_pos += 1
                ret_labels.append([0, 1])
        if not positive_only:
            for i in xrange(N_pos):
                while True:
                    try:
                        anno = self.dataset.propose_negative_Ta()
                        e = Event(self.dataset.games[anno['gameid']]['events'][anno['eid']], gameid=anno['gameid'], anno=anno)
                        e.sequence_around_t(anno, self.dataset.tfr)
                        if extract:
                            # ExtractorException
                            ret_val.append(self.extractor.extract(e))
                        else:
                            # just to make sure event not malformed (like
                            # missing player)
                            _ = self.extractor.extract_raw(e)
                            ret_val.append(e)
                    except EventException as exc:
                        pass
                    except ExtractorException as exc:
                        pass
                    except TeamNotFoundException as exc:
                        pass
                    else:
                        ret_labels.append([1, 0])
                        break
        self.extractor.augment = True
        return np.array(ret_val), np.array(ret_labels)

    def load_train(self, extract=True, positive_only=False):
        return self.load_split(split='train', extract=extract, positive_only=positive_only)

    def load_valid(self, extract=True, positive_only=False):
        return self.load_split(split='val', extract=extract, positive_only=positive_only)

    def _load_event(self, anno, extract, every_K_frame, dont_resolve_basket=False):
        ret_val = []
        ret_gameclocks = []
        ret_frame_idx = []
        event_id = anno['eid']
        game_id = anno['gameid']
        try:
            e = Event(self.dataset.games[game_id]['events'][event_id], gameid=game_id, anno=anno)
        except TeamNotFoundException:
            # malformed event
            return 0
        N_moments = len(e.moments)
        for i in xrange(0, N_moments, every_K_frame):
            try:
                e = Event(self.dataset.games[game_id]['events'][event_id], gameid=game_id, anno=anno)
                game_clock = e.moments[i].game_clock
                quarter = e.moments[i].quarter
                anno = self.dataset._make_annotation(game_id, quarter, game_clock, event_id, self.dataset.tfr)
                e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                # anno = {'gameclock': game_clock}
                # e.sequence_around_t(anno, self.dataset.tfr, from_beginning=False)

                # just to make sure event not malformed (like
                # missing player)
                _ = self.extractor.extract_raw(e)
                ret_val.append(e)
                ret_gameclocks.append(game_clock)
                ret_frame_idx.append(i)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            except TeamNotFoundException as exc:
                pass
        if len(ret_val) == 0:  # malformed Event
            ret = 0
        else:
            if extract:
                ret_val = self.extractor.extract_batch(ret_val,dont_resolve_basket=dont_resolve_basket)
            ret = [ret_val, ret_gameclocks, ret_frame_idx]
        return ret

    def load_split_event(self, split, extract, every_K_frame=4):
        if split == 'val':
            split_hash = self.dataset.val_hash
        elif split == 'train':
            split_hash = self.dataset.train_hash
        elif split == 'game':
            split_hash = self.dataset.game_hash
        else:
            raise NotImplementedError()
        if self.event_index == len(split_hash):
            self.event_index = 0
            return None
        vh = split_hash.values()[self.event_index]

        ret_labels = filter(lambda t: t != -1, [i['gameclock'] for i in vh])
        self.extractor.augment = False
        anno = vh[0]
        ret = self._load_event(anno, extract, every_K_frame)
        if ret == 0:  # malformed Event
            ret = 0
        else:
            ret_val, ret_gameclocks, ret_frame_idx = ret
            meta = [vh[0]['eid'], vh[0]['gameid']]
            ret = [ret_val, ret_labels, ret_gameclocks, meta]
        self.extractor.augment = True
        self.event_index += 1
        return ret

    def load_event(self, game_id, event_id, every_K_frame, player_id=None):
        o = self.extractor.augment
        self.extractor.augment = False
        anno = {"gameid": game_id, "eid": event_id}
        ret = self._load_event(anno, True, every_K_frame, dont_resolve_basket=True)
        self.extractor.augment = o
        return ret

    def reset(self):
        pass


class GameSequenceLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample',
                 fraction_positive=.5, negative_fraction_hard=0):
        """
        """
        self.config = dataset.config
        self.negative_fraction_hard = negative_fraction_hard
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config['preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor
        self.batch_size = batch_size  # not used
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.batch_index = 0

        games = self.dataset.config['data_config']['game_ids']
        self.detect_validation = self.dataset.config['data_config']['detect_validation']

        self.pos_x = []
        self.neg_x = []
        self.pos_t = []
        self.val_x = []
        self.val_t = []

        for game in self.detect_validation:
            if game in games:
                games.remove(game)

        for game in games:
            try:
                pos_x = np.load(os.path.join(self.root_dir, '%s_pos_x.npy' % game))
                neg_x = np.load(os.path.join(self.root_dir, '%s_neg_x.npy' % game))
                pos_t = np.load(os.path.join(self.root_dir, '%s_pos_t.npy' % game))
                val_x = np.load(os.path.join(self.root_dir, '%s_val_x.npy' % game))
                val_t = np.load(os.path.join(self.root_dir, '%s_val_t.npy' % game))
            except IOError:
                continue

            self.pos_x.extend(pos_x)
            self.neg_x.extend(neg_x)
            self.pos_t.extend(pos_t)
            self.val_x.extend(val_x)
            self.val_t.extend(val_t)

        self.val_x = np.array(self.val_x)
        self.val_t = np.array(self.val_t)
        self.pos_x = np.array(self.pos_x)
        self.neg_x = np.array(self.neg_x)
        self.pos_t = np.array(self.pos_t)
        self.neg_t = np.array([[1, 0]]).repeat(self.neg_x.shape[0], axis=0)

        self.pos_ind = 0
        self.neg_ind = 0
        self.val_ind = 0
        self.N_pos = int(batch_size * fraction_positive)
        self.N_neg = batch_size - self.N_pos

        if self.negative_fraction_hard > 0:
            self.N_hard_neg = int(self.N_neg * negative_fraction_hard)
            self.N_neg = self.N_neg - self.N_hard_neg
            self.hard_neg_ind = 0

    def _split(self, inds, fold_index=0):
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(inds)
        N = len(inds)
        val_start = np.round(fold_index/self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1)/self.config['data_config']['N_folds'] * N).astype('int32')
        val_inds= inds[val_start:val_end]
        train_inds = inds[:val_start] + inds[val_end:]
        return train_inds, val_inds

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True):
        # if self.batch_index == self.dataset_size:
        #     return None

        if self.pos_ind + self.N_pos >= self.pos_x.shape[0]:
            self.pos_ind = 0
            self.pos_x, self.pos_t = shuffle_2_array(self.pos_x, self.pos_t)
        if self.neg_ind + self.N_neg >= self.neg_x.shape[0]:
            self.neg_ind = 0
            self.neg_x, self.neg_t = shuffle_2_array(self.neg_x, self.neg_t)
        if (self.negative_fraction_hard > 0 and
                self.hard_neg_ind + self.N_hard_neg >= self.hard_neg_x.shape[0]):
            self.hard_neg_ind = 0
            self.hard_neg_x, self.hard_neg_t = shuffle_2_array(self.hard_neg_x, self.hard_neg_t)

        s = list(self.pos_x.shape)
        s[0] = self.batch_size
        x = np.zeros(s)
        t = np.zeros((self.batch_size, 2))
        x[:self.N_pos] = self.pos_x[self.pos_ind:self.pos_ind + self.N_pos]
        t[:self.N_pos] = self.pos_t[self.pos_ind:self.pos_ind + self.N_pos]
        if not self.negative_fraction_hard > 0:
            x[self.N_pos:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
            t[self.N_pos:] = self.neg_t[self.neg_ind:self.neg_ind + self.N_neg]
        else:
            x[self.N_pos:self.N_pos + self.N_hard_neg] = self.hard_neg_x[
                self.hard_neg_ind:self.hard_neg_ind + self.N_hard_neg]
            t[self.N_pos:self.N_pos + self.N_hard_neg] = self.hard_neg_t[
                self.hard_neg_ind:self.hard_neg_ind + self.N_hard_neg]
            x[self.N_pos + self.N_hard_neg:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
            t[self.N_pos + self.N_hard_neg:] = self.neg_t[self.neg_ind:self.neg_ind + self.N_neg]
        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)
        self.pos_ind += self.N_pos
        self.neg_ind += self.N_neg
        if self.negative_fraction_hard > 0:
            self.hard_neg_ind += self.N_hard_neg
        return x, t

    def load_valid(self, extract=True):
        x = self.val_x
        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)
        t = self.val_t
        return x, t

    def load_test(self, extract=True):
        x, t = [], []
        for game in self.detect_validation:
            pos_x = np.load(os.path.join(self.root_dir, '%s_pos_x.npy' % game))
            pos_t = np.load(os.path.join(self.root_dir, '%s_pos_t.npy' % game))
            neg_x = np.load(os.path.join(self.root_dir, '%s_neg_x.npy' % game))
            neg_t = np.load(os.path.join(self.root_dir, '%s_neg_t.npy' % game))
            val_x = np.load(os.path.join(self.root_dir, '%s_val_x.npy' % game))
            val_t = np.load(os.path.join(self.root_dir, '%s_val_t.npy' % game))

            x.extend(val_x)
            x.extend(pos_x)
            x.extend(neg_x[:len(pos_x)])
            t.extend(val_t)
            t.extend(pos_t)
            t.extend(neg_t[:len(pos_t)])

        inds = range(len(x))
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(inds)

        x = np.array(x)
        t = np.array(t)

        x = x[inds]
        t = t[inds]

        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)

        return x, t

    def reset(self):
        self.batch_index = 0


class TrajectoryLoader:
    def __init__(self, config, fold_index):
        """
        """
        pnr_dir = os.path.join(game_dir, 'pnr-annotations')

        self.config = config
        self.batch_size = self.config['batch_size']
        self.batch_index = 0
        self.fold_index = fold_index
        self.x = []
        self.annotations = []

        self.annotations = pd.read_pickle('%s/roles/annotations.pkl' % (pnr_dir))
        self.x = np.load(open('%s/roles/behaviours.npy' % (pnr_dir), 'rb'))

        self.x = np.array(self.x)
        self.ind = 0
        self.val_ind = 0
        self.set_ind = 0
        self.N = self.batch_size

        train_inds, val_inds = self._split(list(range(len(self.x))))
        self.val_x = self.x[val_inds]
        self.train_x = self.x[train_inds]

    def _split(self, inds, fold_index=0):
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(inds)
        N = len(inds)
        val_start = np.round(fold_index / self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1) / self.config['data_config']['N_folds'] * N).astype('int32')
        val_inds = inds[val_start:val_end]
        train_inds = inds[:val_start] + inds[val_end:]
        return train_inds, val_inds

    def next(self):
        return self.next_batch()

    def next_batch(self):
        if self.ind + self.N >= self.train_x.shape[0]:
            self.ind = 0
            np.random.shuffle(self.train_x)

        try:
            s = list(self.train_x.shape)
        except Exception as err:
            print(None)
        s[0] = self.batch_size
        x = np.zeros(s)
        x[:self.N] = self.train_x[self.ind:self.ind + self.N]
        self.ind += self.N
        return x

    def load_valid(self):
        if self.val_ind + self.batch_size > len(self.val_x):
            self.val_ind = 0
            return None
        x = self.val_x[self.val_ind:self.val_ind + self.batch_size]
        self.val_ind += self.batch_size
        return x

    def load_set(self):
        if self.set_ind + self.batch_size > len(self.x):
            self.set_ind = 0
            return None
        x = self.x[self.set_ind:self.set_ind + self.batch_size]
        annotations = self.annotations[self.set_ind:self.set_ind + self.batch_size]
        self.set_ind += self.batch_size
        return x, annotations

    def reset(self):
        self.batch_index = 0