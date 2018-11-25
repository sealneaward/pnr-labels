"""model_topics.py

Usage:
    model_topics.py <f_data_config>

Arguments:
    <f_data_config>  example ''pnrs.yaml''

Example:
    python model_topics.py pnrs.yaml
"""
from docopt import docopt
import os
import pandas as pd
import json
import numpy as np
import yaml
from copy import copy
import cPickle as pkl

from sklearn.preprocessing import LabelBinarizer as Binarizer
from sklearn.decomposition import LatentDirichletAllocation

import pnr.config as CONFIG


def mold_baseline_vectors(annotations):
    """
    Use actions identified in clusters as vectors to represent as text document.
    Use standard topic modelling techniques to find pnr topic.

    Parameters
    ----------
    annotations: list of dict
        information on pnr annotations

    Returns
    -------
    senctences: list of vectors
        annotation information, as well as topics identified, but with context
    """
    annotations = pd.DataFrame(annotations)
    annotations = annotations[['ball_handler', 'ball_defender', 'screen_setter', 'screen_defender']]

    players = []
    for column in annotations.columns:
        role_players = annotations[column].drop_duplicates(inplace=False).values
        for player in role_players:
            if player not in players:
                players.append(player)

    vectors = []
    for ind, annotation in annotations.iterrows():
        vector = []
        for annotation_player in annotation.values:
            for player in players:
                if annotation_player == player:
                    vector.append(1)
                else:
                    vector.append(0)
        vectors.append(vector)
    vectors = np.array(vectors)
    return vectors

def mold_sentences(vectors):
    """
    Use actions identified in clusters as vectors to represent as text document.
    Use standard topic modelling techniques to find pnr topic.

    Parameters
    ----------
    vectors: dict
        information on pnr annotation, as well as actions identified

    Returns
    -------
    senctences: dict
        annotation information, as well as topics identified, but with context
    """
    annotations = []
    vector_ids = vectors.keys()
    vectorizer = Binarizer()
    vector_sentences = pd.DataFrame()

    for vector_id in vector_ids:
        vector = vectors[vector_id]
        sentence = vector['actions']
        annotation = vector['annotation']

        vector_sentence = pd.DataFrame()
        vector_sentence['id'] = 0
        vector_sentence['player_1_action'] = 0
        vector_sentence['player_2_action'] = 0

        if len(sentence) == 8:
            before_actions = sentence[:4]
            for ind, action in enumerate(before_actions):
                player_vector = {}
                player_vector['player_1_action'] = action
                before_comparisons = [x for i,x in enumerate(before_actions) if i != ind]
                for before_comparison in before_comparisons:
                    player_vector['player_2_action'] = before_comparison
                    player_vector['id'] = vector_id
                    vector_sentence = vector_sentence.append(player_vector, ignore_index=True)

            after_actions = sentence[4:]
            for ind, action in enumerate(after_actions):
                player_vector = {}
                player_vector['player_1_action'] = action
                after_comparisons = [x for i,x in enumerate(after_actions) if i != ind]
                for after_comparison in after_comparisons:
                    player_vector['player_2_action'] = after_comparison
                    player_vector['id'] = vector_id
                    vector_sentence = vector_sentence.append(player_vector, ignore_index=True)

            vector_sentences = vector_sentences.append(vector_sentence)
            annotations.append(annotation)

    vector_sentences['pairwise_actions'] = vector_sentences['player_1_action'].map(str) + vector_sentences['player_2_action'].map(str)
    pairwise_actions = vector_sentences['pairwise_actions']
    pairwise_actions = copy(pd.get_dummies(pairwise_actions))
    pairwise_actions['id'] = vector_sentences['id'].values

    return pairwise_actions, annotations


def find_topics(sentences, annotations, exp_name, n_components=4):
    """
    Use actions identified in clusters as vectors to represent as text document.
    Use standard topic modelling techniques to find pnr topic.

    Parameters
    ----------
    sentences: dict
        information on pnr annotation, as well as actions identified

    Returns
    -------
    topics: dict
        annotation information, as well as topics identified
    """
    vectors = []

    n_actions = sentences.shape[1] - 1
    vocab = list(range(0, n_actions))

    sentences = sentences.groupby('id')
    for sentence_id, sentence in sentences:
        vocab_count = np.zeros(len(vocab))
        for ind, action in sentence.iterrows():
            action.drop('id', inplace=True)
            action = action.values
            action_id = np.argmax(action)

            if action[action_id] > 0:
                vocab_count[action_id] += 1

        vectors.append(vocab_count)

    vectors = np.array(vectors)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0
    )
    lda.fit(vectors)
    topic_probs = lda.transform(vectors)
    for ind, prob in enumerate(topic_probs):
        topic = np.argmax(prob)
        annotations[ind]['topic'] = topic

    pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/%s.pkl' % exp_name), 'wb'))


if __name__ == '__main__':
    from pnr.data.constant import sportvu_dir, game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))
    vectors = pkl.load(open(os.path.join(pnr_dir, 'roles/vectors.pkl'), 'rb'))
    vectors, annotations = mold_sentences(vectors)
    baseline_vectors = mold_baseline_vectors(annotations)
    find_topics(sentences=vectors, annotations=annotations, exp_name='topics')
    # find_topics(sentences=baseline_vectors, annotations=annotations, exp_name='topics-baseline')