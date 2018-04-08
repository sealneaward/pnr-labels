"""train_traj2vec.py

Usage:
    train_traj2vec.py <fold_index> <f_data_config> <f_model_config>

Arguments:
    <f_data_config>  example ''data/config/pnrs.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python train_traj2vec.py 0 pnrs.yaml conv2d-3layers-25x25.yaml
Options:
    --negative_fraction_hard=<percent> [default: 0]
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

def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay):
    loader = TrajectoryLoader(config=data_config, fold_index=fold_index)

    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # TODO update a lot of these placeholders
    learning_rate = tf.placeholder(tf.float32, [])

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    # reporting
    loss = net.loss
    tf.summary.scalar('Reconstruction Loss', loss)

    # checkpoints
    if not os.path.exists(CONFIG.saves.dir):
        os.mkdir(CONFIG.saves.dir)
    # tensorboard
    if not os.path.exists(CONFIG.logs.dir):
        os.mkdir(CONFIG.logs.dir)

    # remove existing log folder for the same model.
    # if os.path.exists(CONFIG.logs.dir):
    #     import shutil
    #     shutil.rmtree(CONFIG.logs.dir)

    train_writer = tf.summary.FileWriter(os.path.join(CONFIG.logs.dir, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(CONFIG.logs.dir, 'val'), sess.graph)
    merged = tf.summary.merge_all()

    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    best_eval_loss = np.inf

    tf.global_variables_initializer().run()

    best_not_updated = 0
    lrv = init_lr

    # Train
    for iter_ind in tqdm(range(max_iter)):
        best_not_updated += 1
        loaded = loader.next()

        if loaded is not None:
            batch_xs = loaded
        else:
            loader.reset()
            continue

        if iter_ind % 250000 == 0 and iter_ind > 0:
            lrv *= .1

        feed_dict = net.input(batch_xs)
        feed_dict[learning_rate] = lrv
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, iter_ind)

        if iter_ind % 1000 == 0:
            feed_dict = net.input(batch_xs)
            train_loss = loss.eval(feed_dict=feed_dict)

            tf_eval_loss = []
            while True:
                loaded = loader.load_set()
                if loaded is not None:
                    eval_x, _ = loaded
                    feed_dict = net.input(eval_x)
                    eval_loss = sess.run([loss], feed_dict=feed_dict)
                    tf_eval_loss.append(eval_loss)
                else:  ## done
                    eval_loss = sess.run([loss], feed_dict=feed_dict)
                    tf_eval_loss.append(eval_loss)
                    eval_loss = np.mean(tf_eval_loss)
                    break

            print("Step %d, Training Loss %g, Evaluation Loss %g" % (iter_ind, train_loss, eval_loss))

            if eval_loss < best_eval_loss:
                best_not_updated = 0
                p = '%s/%s.ckpt.best' % (CONFIG.saves.dir, exp_name)
                print ('Saving Best Model to: %s' % p)
                best_saver.save(sess, p)
                tf.train.export_meta_graph('%s.meta' % (p))
                best_eval_loss = eval_loss


        if best_not_updated == best_acc_delay:
            break


if __name__ == '__main__':
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
    max_iter = 500000
    best_acc_delay = 50000
    train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay)
