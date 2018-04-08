import tensorflow as tf
import numpy as np

from pnr.model.utils import lstm_cell

class Seq2Seq:
    """
    seq2seq model for trajectory clustering
    """

    def __init__(self, config):
        self.config = config
        self.input_length = self.config['input_length']
        self.batch_size = self.config['batch_size']
        self.hidden_dim = self.config['hidden_dim']
        self.feature_size = self.config['feature_size']
        self.keep_prob = self.config['keep_prob']
        self.num_rnn = self.config['num_rnn']

    def loopf(self, prev, _):
        return prev

    def build(self):

        # the sequences, has n steps of maximum size
        seq_input = tf.placeholder(tf.float32, [self.input_length, self.batch_size, self.feature_size], name='seq_input')
        # what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]

        # inputs for rnn needs to be a list, each item/frame being a timestep.
        # we need to split our input into each timestep, and reshape it because split keeps dims by default
        loss_inputs = [tf.reshape(seq_input, [-1])]
        # encoder_inputs B, T, F
        encoder_inputs = [item for item in tf.unstack(seq_input)]
        # if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
        # and target size equal encoder size plus 1. For simplicity, here I droped the last one.
        # decoder_inputs B, T, F
        decoder_inputs = ([tf.zeros_like(encoder_inputs[0])] + encoder_inputs[:-1])

        # basic LSTM seq2seq model
        cell = tf.nn.rnn_cell.MultiRNNCell([
            lstm_cell(self.hidden_dim, self.keep_prob) for _ in range(self.num_rnn)
        ])
        self.outputs, self.enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.feature_size)
        dec_outputs, self.dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs,
            self.enc_state,
            cell,
            loop_function=self.loopf
        )

        # flatten the prediction and target to compute squared error loss
        y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
        y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

        # define loss and optimizer, minimize the squared error
        self.loss = 0
        for i in range(len(loss_inputs)):
            self.loss += tf.reduce_sum(tf.square(tf.subtract(y_pred[i], y_true[len(loss_inputs) - i - 1])))

        # define input for feed_dict
        self.x = seq_input

    def input(self, x):
        x = self.transpose(x)
        ret_dict = {}
        ret_dict[self.x] = x

        return ret_dict

    def transpose(self, x):
        x = np.transpose(x, (1, 0, 2))
        return x