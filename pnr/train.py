"""train.py

Usage:
    train.py <fold_index> <f_data_config> <f_model_config>

Arguments:
    <f_data_config>  example ''data/config/pnrs.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python train.py 0 pnrs.yaml conv2d-3layers-25x25.yaml
Options:
    --negative_fraction_hard=<percent> [default: 0]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# model
import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss

from pnr.model.convnet2d import ConvNet2d
from pnr.data.dataset import BaseDataset
from pnr.data.extractor import BaseExtractor
from pnr.data.loader import *
import pnr.config as CONFIG

from tqdm import tqdm
from docopt import docopt
import yaml
import numpy as np

def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, fold_index, load_raw=False, no_anno=True)
    extractor = BaseExtractor(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
        pfh = 1 - nfh
    else:
        nfh = 0
        pfh = 0.5
    loader = GameSequenceLoader(
        dataset,
        extractor,
        data_config['batch_size'],
        fraction_positive=pfh,
        negative_fraction_hard=nfh
    )

    val_x, val_t = loader.load_valid()

    if model_config['class_name'] == 'ConvNet2d':
        val_x = np.rollaxis(val_x, 1, 4)
    else:
        raise Exception('input format not specified')

    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    y_ = tf.placeholder(tf.float32, [None, dataset.n_classes])
    weights = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * 0.001
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
    loss = tf.reduce_mean(cross_entropy + l2_loss)

    # optimize
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    global_step = tf.Variable(0)
    learning_rate = tf.placeholder(tf.float32, [])
    training = tf.placeholder(tf.bool)
    # learning_rate = tf.train.exponential_decay(init_lr, global_step, 10000, 0.96, staircase=True)
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate,
    #             optimizer=lambda lr: tf.train.AdamOptimizer(lr))
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate,
    #             optimizer=lambda lr: tf.train.MomentumOptimizer(lr, .9))
    # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
    train_step = optimize_loss(cross_entropy, global_step, learning_rate, optimizer=lambda lr: tf.train.AdamOptimizer(lr, .9))

    # reporting
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predictions = tf.argmax(net.output(), 1)
    true_labels = tf.argmax(y_, 1)

    tf.summary.histogram('label_distribution', y_)
    tf.summary.histogram('logits', net.logits)
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)


    # checkpoints
    if not os.path.exists(CONFIG.saves.dir):
        os.mkdir(CONFIG.saves.dir)
    # tensorboard
    if not os.path.exists(CONFIG.logs.dir):
        os.mkdir(CONFIG.logs.dir)

    if dataset.n_classes == 2:
        confusion_matrix = tf.confusion_matrix(labels=true_labels, predictions=predictions, num_classes=2)
        tp = tf.count_nonzero(predictions * true_labels)
        tn = tf.count_nonzero((predictions - 1) * (true_labels - 1))
        fp = tf.count_nonzero(predictions * (true_labels - 1))
        fn = tf.count_nonzero((predictions - 1) * true_labels)

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('F-Score', fmeasure)


    # if model_config['class_name'] == 'ConvNet2d' or model_config['class_name'] == 'ConvNet2dDeep':
    #     tf.summary.image('input', net.x, max_outputs=5)
    # elif model_config['class_name'] == 'ConvNet3d' or model_config['class_name'] == 'LSTM':
    #     tf.summary.image('input', tf.reduce_sum(net.x, 1), max_outputs=4)
    # else:
    #     raise Exception('input format not specified')

    merged = tf.summary.merge_all()
    log_folder = '%s/%s' % (CONFIG.logs.dir,exp_name)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder)

    train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
    tf.global_variables_initializer().run()
    # Train
    # best_val_acc = 0
    best_val_ce = np.inf
    best_not_updated = 0
    lrv = init_lr
    for iter_ind in tqdm(range(max_iter)):
        best_not_updated += 1
        # if iter_ind ==0:
        loaded = loader.next()
        if loaded is not None:
            batch_xs, batch_ys = loaded
        else:
            loader.reset()
            continue
        if model_config['class_name'] == 'ConvNet2d':
            batch_xs = np.rollaxis(batch_xs, 1, 4)
        else:
            raise Exception('input format not specified')

        feed_dict = net.input(batch_xs, None, True)
        feed_dict[y_] = batch_ys

        if iter_ind % 10000 == 0 and iter_ind > 0:
            lrv *= .1
        feed_dict[learning_rate] = lrv
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, iter_ind)
        if iter_ind % 100 == 0:
            feed_dict = net.input(batch_xs, 1, False)
            feed_dict[y_] = batch_ys
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            train_ce = cross_entropy.eval(feed_dict=feed_dict)
            # train_confusion_matrix = confusion_matrix.eval(feed_dict=feed_dict)

            feed_dict = net.input(val_x, 1, False)
            feed_dict[y_] = val_t
            summary, val_ce, val_accuracy = sess.run([merged, cross_entropy, accuracy], feed_dict=feed_dict)
            val_writer.add_summary(summary, iter_ind)
            print("step %d, training accuracy %g, validation accuracy %g, \n train ce %g,  val ce %g" %
                  (iter_ind, train_accuracy, val_accuracy, train_ce, val_ce))

            if val_ce < best_val_ce:
                best_not_updated = 0
                p = '%s/%s.ckpt.best' % (CONFIG.saves.dir, exp_name)
                print ('Saving Best Model to: %s' % p)
                save_path = best_saver.save(sess, p)
                tf.train.export_meta_graph('%s.meta' % (p))
                best_val_ce = val_ce
        if iter_ind % 2000 == 0:
            save_path = saver.save(sess,'%s/%s-%d.ckpt'%(CONFIG.saves.dir,exp_name,iter_ind))
        if best_not_updated == best_acc_delay:
            break
    return best_val_ce


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
    max_iter = 30000
    best_acc_delay = 10000
    train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay)