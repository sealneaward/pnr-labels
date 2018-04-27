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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pnr.config as CONFIG


def find_topics(sentences, data_config, n_actions=20, n_components=4):
    """
    Use actions identified in clusters as vectors to represent as text document.
    Use standard topic modelling techniques to find pnr topic.

    Parameters
    ----------
    sentences: dict
        information on pnr annotation, as well as actions identified
    data_config: yaml dict
        data configuration information

    Returns
    -------
    topics: dict
        annotation information, as well as topics identified
    """
    vectorizer = CountVectorizer(min_df=0)
    vectors = []
    annotations = []

    senctence_ids = sentences.keys()
    vocab = list(range(-1, n_actions))

    for sentence_id in senctence_ids:
        vocab_count = np.zeros(len(vocab))
        sentence = sentences[sentence_id]
        annotation = sentence['annotation']
        sentence = sentence['actions']

        for action in sentence:
            vocab_count[action + 1] += 1

        vectors.append(vocab_count)
        annotations.append(annotation)

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

    pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/topics.pkl'), 'wb'))



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
    find_topics(vectors, data_config)