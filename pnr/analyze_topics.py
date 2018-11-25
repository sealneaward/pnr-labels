"""analyze_topics.py

Usage:
    analyze_topics.py <f_data_config>

Arguments:
    <f_data_config>  example ''pnrs.yaml''

Example:
    python analyze_topics.py pnrs.yaml
"""
from docopt import docopt
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

import pnr.config as CONFIG

def read_extended(dir_path, topics):
    """
    Get the extended set of annotations for evaluation
    """
    all_raw_f = filter(lambda s: 'raw-' in s, os.listdir(dir_path))
    annotations = pd.DataFrame()
    topics = pd.DataFrame(topics)
    topics['gameid'] = '00' + topics['gameid'].astype(int).astype(str)

    for game_f in all_raw_f:
        game_id = game_f.split('-')[-1].split('.')[0]
        game_annotations = pd.read_csv('%s/%s' % (dir_path, game_f))
        game_annotations['gameid'] = game_id
        annotations = annotations.append(game_annotations)

    # limit to test answers predicted
    annotations = pd.merge(left=topics, right=annotations, on=['gameid', 'eid', 'gameclock', 'quarter'], how='inner')
    annotations = annotations.drop('topic', inplace=False, axis=1)
    topics = pd.merge(left=annotations, right=topics, on=['gameid', 'eid', 'gameclock', 'quarter'], how='inner')
    topics = topics.drop(['over', 'under', 'switch', 'trap'], inplace=False, axis=1)
    print('Shape annotations: %s' % str(annotations.shape))
    print('Shape topics: %s' % str(topics.shape))

    # encode one hot to categorical
    annotations[['over', 'under', 'switch', 'trap']] = annotations[['over', 'under', 'switch', 'trap']].fillna(0)
    annotations[['over', 'under', 'switch', 'trap']] = annotations[['over', 'under', 'switch', 'trap']].replace('x', 1)

    return annotations, topics


def evaluate_topics(topics, annotations, data_config):
    """

    Parameters
    ----------

    Returns
    -------
    """
    topic_config = data_config['topics']
    topics['topic'] = topics['topic'].map(topic_config)
    topics = pd.get_dummies(topics['topic'])

    predictions = topics[['over', 'under', 'switch', 'trap']].values
    y_true = annotations[['over', 'under', 'switch', 'trap']].values

    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_true, axis=1)

    print(confusion_matrix(y_pred=predictions, y_true=y_true))
    print(classification_report(y_pred=predictions, y_true=y_true))


if __name__ == '__main__':
    from pnr.data.constant import sportvu_dir, game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    topics = pd.read_pickle('%s/roles/topics.pkl' % (pnr_dir))
    # topics = pd.read_pickle('%s/roles/topics-baseline.pkl' % (pnr_dir))
    annotations, topics = read_extended('%s/extended' % (pnr_dir), topics)

    evaluate_topics(topics, annotations, data_config)