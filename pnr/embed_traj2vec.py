"""embed_traj2vec.py

Usage:
    embed_traj2vec.py <fold_index> <f_data_config> <f_model_config>

Arguments:
    <f_data_config>  example ''data/config/pnrs.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python embed_traj2vec.py 0 pnrs.yaml conv2d-3layers-25x25.yaml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# model
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
optimize_loss = tf.contrib.layers.optimize_loss

from pnr.model.seq2seq import Seq2Seq
import pnr.config as CONFIG
from pnr.data.loader import TrajectoryLoader

from tqdm import tqdm
from docopt import docopt
import yaml
import numpy as np
import os
import cPickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

def embed(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay):
    # hard code batch_size to 1 for writing embeddings
    data_config['batch_size'] = 1
    model_config['model_config']['batch_size'] = 1

    loader = TrajectoryLoader(config=data_config, fold_index=fold_index)
    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # reporting
    loss = net.loss
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    ckpt_path = '%s/%s.ckpt.best' % (CONFIG.saves.dir, exp_name)
    saver.restore(sess, ckpt_path)

    embeddings = []
    embeddings_annotations = []

    while True:
        # save embedding
        loaded = loader.load_set()
        if loaded is not None:
            x, annotations = loaded
            feed_dict = net.input(x)
            embedding = sess.run([net.enc_state], feed_dict=feed_dict)
            embedding = np.array(embedding)
            embedding = np.squeeze(embedding)
            embedding = embedding[0]
            # get last embedded state
            embeddings.append(embedding)
            embeddings_annotations.append(annotations[0])
        else:  ## done
            embeddings = np.array(embeddings)
            np.save(os.path.join(pnr_dir, 'roles/embeddings'), embeddings)
            embeddings = pd.DataFrame(data=embeddings, columns=['%s' % col for col in range(embeddings.shape[-1])])
            embeddings.to_csv('%s/%s' % (os.path.join(pnr_dir, 'roles'), 'embeddings.csv'), index=False, sep='\t')
            pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/embeddings_annotations.pkl'), 'wb'))
            break

    return embeddings

def project_tf(embeddings):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    embedding_var = tf.Variable(tf.stack(embeddings, axis=0), trainable=False, name='embedding')
    sess.run(embedding_var.initializer)
    writer = tf.summary.FileWriter('%s/%s' % (CONFIG.logs.dir, 'projector'), sess.graph)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # # Comment out if you don't have metadata
    # embedding.metadata_path = os.path.join(CONFIG.logs.dir, 'metadata.tsv')
    #
    # # Comment out if you don't want sprites
    # embedding.sprite.image_path = os.path.join(CONFIG.logs.dir, 'sprite.png')

    projector.visualize_embeddings(writer, config)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(CONFIG.logs.dir, 'embed.ckpt'), 1)
    np.save('%s/embeddings/embeddings' % (sportvu_dir), embeddings)


if __name__ == '__main__':
    from pnr.data.constant import game_dir, sportvu_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")
    f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
    f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])


    data_config = yaml.load(open(f_data_config, 'rb'))
    model_config = yaml.load(open(f_model_config, 'rb'))
    model_name = os.path.basename(f_model_config).split('.')[0]
    data_name = os.path.basename(f_data_config).split('.')[0]
    exp_name = '%s-X-%s' % (model_name, data_name)
    fold_index = int(arguments['<fold_index>'])
    init_lr = 1e-3
    max_iter = 30000
    best_acc_delay = 10000
    embeddings = embed(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay)
    # tsne(embeddings)
    project_tf(embeddings)