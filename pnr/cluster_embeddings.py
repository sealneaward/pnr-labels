"""cluster_embeddings.py

Usage:
    cluster_embeddings.py

Example:
    python cluster_embeddings
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

import pnr.config as CONFIG

def clean_data(data):
    data = data[0]
    clean_data = []
    for projection in data['projections']:
        clean_data.append([projection['tsne-0'], projection['tsne-1'], projection['tsne-2']])
    clean_data = np.array(clean_data)
    return clean_data

def cluster(data):
    """
    Use K-Means in 3D space to find best number of cluster for action types.

    Parameters
    ----------
    data: np.array
        3d location points
    """
    X = StandardScaler().fit_transform(data[:, :2])

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.1, min_samples=20).fit(X)
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

if __name__ == '__main__':
    from pnr.data.constant import sportvu_dir

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    data = json.load(open('%s/embeddings/%s' % (sportvu_dir, 'embedding_actions.txt'), 'r'))
    data = clean_data(data)
    cluster(data)