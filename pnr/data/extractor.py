from __future__ import division
import cPickle as pickle
import yaml
import os

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pnr.data.utils import (
    pictorialize_team, pictorialize_fast, make_3teams_11players,
    make_reference, scale_last_dim, make_reference_pnr, pictorialize_fast_pnr
)
from pnr.data.constant import game_dir


class ExtractorException(Exception):
    pass

class OneHotException(Exception):
    pass

class BaseExtractor(object):
    """base class for sequence extraction
        Input: a truncated Event
        Output: classifier model input

    Simplest possible parametrization, a collapsed image of the full court
    Options (all 3 can be used in conjunction):
        -d0flip
        -d1flip
        -jitter (x,y)
    returns a 3 channel image of (ball, offense, defense)
            -> resolves possession: which side of the court it's one
    """

    def __init__(self, f_config):
        self.augment = True
        if type(f_config) == str:
            self.config = yaml.load(open(f_config, 'rb'))['extractor_config']
        else:
            self.config = f_config['extractor_config']

    def extract_raw(self, event, dont_resolve_basket=False):
        """
        """
        ##
        moments = event.moments
        off_is_home = event.is_home_possession(moments[len(moments) // 2])
        ball, offense, defense = [[]], [[], [], [], [], []], [[], [], [], [], []]
        for moment in moments:
            ball[0].append([moment.ball.x, moment.ball.y])
            off_id, def_id = 0, 0
            for player_idx, player in enumerate(moment.players):
                if dont_resolve_basket:
                    if player_idx < 5:
                        offense[off_id].append([player.x, player.y])
                        off_id += 1
                    else:
                        defense[def_id].append([player.x, player.y])
                        def_id += 1
                else:
                    if (player.team.id == event.home_team_id) == off_is_home:  # offense
                        offense[off_id].append([player.x, player.y])
                        off_id += 1
                    else:  # defense
                        defense[def_id].append([player.x, player.y])
                        def_id += 1
        if (
                len(ball) == 0 or
                (
                        not (
                            (len(np.array(ball).shape) == 3 and len(np.array(offense).shape) == 3
                            and len(np.array(defense).shape) == 3)
                            and (
                                np.array(ball).shape[1] == np.array(offense).shape[1]
                                and np.array(offense).shape[1] == np.array(defense).shape[1]
                            )
                        )
                )
        ):
            raise ExtractorException()

        return [ball, offense, defense]

    def extract_raw_pnr(self, event, one_hot=False):
        """
        """
        ##
        moments = event.moments
        off_is_home = event.is_home
        ball, ball_player, ball_defense_player, screen_player, screen_defense_player = [], [], [], [], []
        ball_player_vector, ball_defense_player_vector, screen_player_vector, screen_defense_player_vector = None, None, None, None

        try:
            for moment in moments:
                ball.append([moment.ball.x, moment.ball.y])
                for player_idx, player in enumerate(moment.players):
                    if player_idx == 0:
                        ball_player.append([player.x, player.y])
                    elif player_idx == 1:
                        screen_player.append([player.x, player.y])
                    elif player_idx == 2:
                        ball_defense_player.append([player.x, player.y])
                    elif player_idx == 3:
                        screen_defense_player.append([player.x, player.y])
        except Exception:
            raise ExtractorException

        if one_hot:
            for player_idx, player in enumerate(moment.players):
                if player_idx == 0:
                    ball_player_vector = player.one_hot_vector
                elif player_idx == 1:
                    screen_player_vector = player.one_hot_vector
                elif player_idx == 2:
                    ball_defense_player_vector = player.one_hot_vector
                elif player_idx == 3:
                    screen_defense_player_vector = player.one_hot_vector


        if(
            len(ball) == 0 or
            (
                not (
                    len(np.array(ball).shape) == 2 and
                    len(np.array(ball_player).shape) == 2 and
                    len(np.array(ball_defense_player).shape) == 2 and
                    len(np.array(screen_player).shape) == 2 and
                    len(np.array(screen_defense_player).shape) == 2
                )
            )
        ):
            raise ExtractorException()
        if not one_hot:
            return [ball, ball_player, screen_player, ball_defense_player, screen_defense_player]
        else:
            if (
                not (
                    ball_player_vector is not None and
                    screen_player_vector is not None and
                    ball_defense_player_vector is not None and
                    screen_defense_player_vector is not None
                )
            ):
                raise OneHotException()

            return [
                ball,
                ball_player,
                screen_player,
                ball_defense_player,
                screen_defense_player
            ], [
                ball_player_vector,
                screen_player_vector,
                ball_defense_player_vector,
                screen_defense_player_vector
            ]

    def extract(self, event):
        sample_rate = 1
        Y_RANGE = 100
        X_RANGE = 50
        x = self.extract_raw(event)
        ctxy = []
        if self.augment and np.sum(self.config['jitter']) > 0:
            d0_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][0]
            d1_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][1]
            jit = np.array([d0_jit, d1_jit])
            jit = jit.reshape(1, 2).repeat(len(x[0][0]), axis=0)
            for team in x:
                for player in team:
                    try:
                        player = np.array(player) + jit
                    except ValueError:  # bad sequence where not all players have the same number of moments
                        raise ExtractorException()

        for play_sequence in x:
            try:
                team_matrix = np.concatenate(play_sequence, 1)
            except ValueError:
                raise ExtractorException()

            tm = pictorialize_team(team_matrix, sample_rate=sample_rate,
                                   Y_RANGE=Y_RANGE, X_RANGE=X_RANGE)

            ctxy.append(tm)
        ctxy = np.array(ctxy)
        if len(ctxy.shape) == 1:  # different teams have different length
            raise ExtractorException()
        # compress the time dimension
        if 'video' in self.config and self.config['video']:
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                ctxy = ctxy[:, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                ctxy = ctxy[:, :, :, ::-1]
            return ctxy
        else:
            cxy = ctxy.sum(1)
            cxy[cxy > 1] = 1
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                cxy = cxy[:, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                cxy = cxy[:, :, ::-1]
            return cxy

    def extract_batch(self, events_arr, input_is_sequence=False, dont_resolve_basket=False):
        sample_rate = 1
        Y_RANGE = 100
        X_RANGE = 50
        if input_is_sequence:
            sequences = events_arr
        else:
            if self.config['pnr']:
                sequences = np.array([
                        self.extract_raw_pnr(e) for e in events_arr
                ])
            else:
                sequences = np.array([
                    make_3teams_11players(
                        self.extract_raw(e, dont_resolve_basket=dont_resolve_basket)
                    ) for e in events_arr
                ])
        # time crop (+jitter) , spatial crop
        if 'version' in self.config and self.config['version'] >= 2:
            if self.augment:
                t_jit = np.min([self.config['tfa_jitter_radius'], sequences.shape[2] / 2 - self.config['tfr']])
                t_jit = (2 * t_jit * np.random.rand()).round().astype('int32') - t_jit
            else:
                t_jit = 0
            tfa = int(sequences.shape[2] / 2 + t_jit)
            sequences = sequences[:, :, tfa - self.config['tfr']:tfa + self.config['tfr']]
            if 'crop' in self.config and self.config['crop'] != '':
                reference = make_reference(sequences, self.config['crop_size'], self.config['crop'])
                sequences = sequences - reference
                Y_RANGE = self.config['crop_size'][0] + 2
                X_RANGE = self.config['crop_size'][1] + 2
        # spatial jitter
        if self.augment and np.sum(self.config['jitter']) > 0:
            d0_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][0]
            d1_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][1]
            # hacky: can delete after -- temporary for malformed data (i.e.
            # missing player)
            try:
                sequences[:, :, :, 0] += d0_jit
            except:
                raise ExtractorException()
            sequences[:, :, :, 1] += d1_jit
        ##
        bctxy = pictorialize_fast(sequences, sample_rate, Y_RANGE, X_RANGE)

        # if cropped, shave off the extra padding
        if ('version' in self.config and self.config['version'] >= 2 and 'crop' in self.config):
            bctxy = bctxy[:, :, :, 1:-1, 1:-1]
        # compress the time dimension
        if 'video' in self.config and self.config['video']:
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                bctxy = bctxy[:, :, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                bctxy = bctxy[:, :, :, :, ::-1]
            return bctxy
        else:
            bcxy = bctxy.sum(2)
            bcxy[bcxy > 1] = 1
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                bcxy = bcxy[:, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                bcxy = bcxy[:, :, :, ::-1]

            # debugging
            # for img in bcxy:
            #     for ind, player in enumerate(img):
            #         plt.subplot(2, 3, ind + 1)
            #         plt.imshow(player)
            #     plt.show()
            #     plt.close()

            return bcxy
