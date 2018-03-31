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

def embed(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay):
    # hard code batch_size to 1 for writing embeddings
    # data_config['batch_size'] = 1
    # model_config['model_config']['batch_size'] = 1

    loader = TrajectoryLoader(config=data_config, fold_index=fold_index)
    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # reporting
    loss = net.loss
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    ckpt_path = '%s/%s.ckpt.best' % (CONFIG.saves.dir, exp_name)
    saver.restore(sess, ckpt_path)

    # get last val loss figure
    val_tf_loss = []
    while True:
        loaded = loader.load_valid()
        if loaded is not None:
            val_x = loaded
            feed_dict = net.input(val_x)
            val_loss = sess.run([loss], feed_dict=feed_dict)
            val_tf_loss.append(val_loss)
        else:  ## done
            val_loss = sess.run([loss], feed_dict=feed_dict)
            val_tf_loss.append(val_loss)
            val_loss = np.mean(val_tf_loss)
            break

    print("Best Validation Loss %g" % (val_loss))

    embeddings = []
    embeddings_annotations = []

    # while True:
    #     # save embedding
    #     loaded = loader.load_set()
    #     if loaded is not None:
    #         x, annotations = loaded
    #         feed_dict = net.input(x)
    #         embedding = sess.run([net.enc_state], feed_dict=feed_dict)
    #         embedding = np.array(embedding)
    #         # embedding = np.squeeze(embedding)
    #         # get last embedded state
    #         # TODO check if having 5 embedded states has a difference on embedding process
    #         # traj2vec only has 1 embedded state, I have 5  TODO check window logic
    #         embeddings.append(embedding[0, 0, -1])
    #         embeddings_annotations.append(annotations[0])
    #     else:  ## done
    #         embeddings = np.array(embeddings)
    #         np.save(os.path.join(pnr_dir, 'roles/embeddings'), embeddings)
    #         pkl.dump(annotations, open(os.path.join(pnr_dir, 'roles/embeddings_annotations.pkl'), 'wb'))
    #         break


if __name__ == '__main__':
    from pnr.data.constant import game_dir
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
    embed(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay)
