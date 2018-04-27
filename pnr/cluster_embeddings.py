"""cluster_embeddings.py

Usage:
    cluster_embeddings.py <f_data_config>

Arguments:
    <f_data_config>  example ''pnrs.yaml''

Example:
    python cluster_embeddings pnrs.yaml
"""
from docopt import docopt
import os
import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import yaml
from copy import copy
import cPickle as pkl

import pnr.config as CONFIG
from pnr.plots.plot import plot_action

def clean_data(data):
    data = data[0]
    clean_data = []
    for projection in data['projections']:
        clean_data.append([projection['tsne-0'], projection['tsne-1'], projection['tsne-2']])
    clean_data = np.array(clean_data)
    return clean_data

def cluster_db(data):
    """
    Use DBSCAN in 2D space to find best number of cluster for action types.

    Parameters
    ----------
    data: np.array
        3d location points

    Returns
    -------
    labels: np.array
        cluster identification for annotation
    """
    X = StandardScaler().fit_transform(data[:, :2])

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.105, min_samples=40).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    plt.close()

    return db.labels_


def cluster_kmeans(data, n_clusters=20):
    """
    Use KMeans in 3D space to find best number of cluster for action types.

    Parameters
    ----------
    data: np.array
        3d location points

    Returns
    -------
    labels: np.array
        cluster identification for annotation
    """
    X = StandardScaler().fit_transform(data)

    km = KMeans(n_clusters=n_clusters)
    km.fit(X)
    labels = km.labels_

    return labels




def vis_labels(annotations, clusters, data_config):
    """
    Visualize the different actions identified through the encoding process.

    Parameters
    ----------
    annotations: list of dicts
        information about actions, including cluster ids
    clusters: np.array
        action label information
    data_config: dict
        configuration for annotations

    """
    for ind, annotation in enumerate(annotations):
        label = clusters[ind]
        annotation['label'] = label
        annotations[ind] = annotation

    annotations = pd.DataFrame(annotations)
    annotations['gameid'] = '00' + annotations['gameid'].astype(int).astype(str).values
    action_types = annotations['label'].drop_duplicates(inplace=False).values

    action_types = [14]
    for action_type in action_types:
        action_annotations = annotations.loc[annotations.label == action_type, :]
        game_ids = annotations['gameid'].drop_duplicates(inplace=False).values
        for game_id in game_ids:
            game = pd.read_pickle(os.path.join(game_dir, game_id + '.pkl'))
            game_action_annotations = action_annotations.loc[action_annotations.gameid == game_id, :]
            for ind, action in game_action_annotations.iterrows():

                plot_action(game, action, game_id, data_config)


def make_vectors(annotations, clusters, data_config):
    """
    Create a "paragraph vector" of the actions identified.

    Parameters
    ----------
    annotations: list of dicts
        information about actions, including cluster ids
    clusters: np.array
        action label information
    data_config: dict
        configuration for annotations

    Returns
    -------
    sentences: np.array
        information for a single annotation on the 4 roles and the 2 seperate actions they perform in a pnr

    """
    for ind, annotation in enumerate(annotations):
        label = copy(clusters[ind])
        annotation['label'] = label
        annotations[ind] = annotation

    annotations = pd.DataFrame.from_records(annotations)
    unique_annotations = annotations[[
        'gameid',
        'eid',
        'quarter',
        'gameclock',
        'ball_handler',
        'ball_defender',
        'screen_setter',
        'screen_defender'
    ]].drop_duplicates(inplace=False)
    annotations_dict = {}

    for ind, unique_annotation in unique_annotations.iterrows():
        annotations_at_id = annotations.loc[
            (annotations.gameid == unique_annotation['gameid']) &
            (annotations.quarter == unique_annotation['quarter']) &
            (annotations.gameclock == unique_annotation['gameclock'])
        ,:]

        labels = annotations_at_id['label'].values
        annotations_dict[ind] = {}
        annotations_dict[ind]['actions'] = labels
        annotations_dict[ind]['annotation'] = unique_annotation

    pkl.dump(annotations_dict, open(os.path.join(pnr_dir, 'roles/vectors.pkl'), 'wb'))

if __name__ == '__main__':
    from pnr.data.constant import sportvu_dir, game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    annotations = pd.read_pickle('%s/roles/annotations.pkl' % (pnr_dir))
    embeddings = np.load(open('%s/embeddings/embeddings.npy' % (sportvu_dir), 'rb'))
    data = json.load(open('%s/embeddings/%s' % (sportvu_dir, 'embedding_actions.txt'), 'r'))
    data = clean_data(data)
    clusters = cluster_kmeans(data)
    # clusters = cluster_db(embeddings)
    # vis_labels(annotations, clusters, data_config)
    make_vectors(annotations, clusters, data_config)