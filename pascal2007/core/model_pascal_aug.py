# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import numpy as np


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, \
                 n_time_step=8, prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, \
                 dropout=True, batch_size=128):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (2) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (2) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {0:'<NULL>', 1:'<START>', 2:'<END>', 3:'aeroplane', 4:'bicycle', 5:'bird', 6:'boat', 7:'bottle', 8:'bus', \
                9:'car', 10:'cat', 11:'chair', 12:'cow', 13:'diningtable', 14:'dog', 15:'horse', 16:'motorbike', \
                17:'person', 18:'pottedplant', 19:'sheep', 20:'sofa', 21:'train', 22:'tvmonitor'}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.groundtruth = tf.placeholder(tf.float32, [None, self.V])
        self.masks = tf.placeholder(tf.float32, [self.T, None, self.V])
        self.c = tf.placeholder(tf.float32, [1,1024])
        self.h = tf.placeholder(tf.float32, [1,1024])
        self.samp = tf.placeholder(tf.int32, [1])
        self.x = tf.placeholder(tf.float32, [1, len(word_to_idx)])
        '''
        self.batch_features = tf.contrib.layers.batch_norm(
                                            inputs = self.features,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=False,
                                            updates_collections=None,
                                            scope=('conv_features/batch_norm'))
        with tf.variable_scope('init_project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(self.features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
        self.proj_features = features_proj
        '''
    '''
    def set_end_time(self, end_time):
        self.end_time = end_time
    '''
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, x, reuse=False):
        '''
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [batch_size, batch_size*self.M], initializer=self.emb_initializer)
            b = tf.get_variable('b', [batch_size*self.M], initializer=self.const_initializer)
            x = tf.reshape(tf.nn.sigmoid(tf.matmul(inputs, w) + b, 'x'), [batch_size, self.M])
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
        '''
        x += tf.to_float(tf.one_hot(inputs, self.V, on_value=1))
        return x

    def _project_features(self, features, reuse=False):
        with tf.variable_scope('project_features', reuse=reuse):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.mul(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.V], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.V], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.V, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.V], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out # dim: N x V
            return out_logits

    def _batch_norm(self, x, mode='train', name=None, reuse=False):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            reuse=reuse,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        masks = self.masks
        batch_size = tf.shape(features)[0]
        groundtruth = tf.to_float(self.groundtruth)
        groundtruth_mask = tf.zeros([batch_size, self.V], tf.float32)
        groundtruth_mask += groundtruth * 100
        all_ones = tf.ones([batch_size, self.V], tf.float32)

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        start_ind = tf.ones([batch_size], tf.int32)
        x = self._word_embedding(inputs=start_ind, x=tf.zeros([batch_size, self.V], tf.float32))
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, dropout=self.dropout, reuse=(t!=0))
            # logits = tf.Print(logits, [logits], message="logits = ", summarize=10)
            loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, groundtruth) * masks[t])

            # predicted labels and groundtruth_mask
            logits += (groundtruth_mask - all_ones * 100)
            next_ind = tf.argmax(logits, 1)
            # next_ind = tf.Print(next_ind, [next_ind], message="next_ind is: ", summarize=200)
            groundtruth_mask += tf.to_float(tf.one_hot(next_ind, self.V, on_value=-100))

            # modify groundtruth
            # groundtruth += tf.to_float(tf.one_hot(next_ind, self.V, on_value=-1))

            # next label
            x = self._word_embedding(inputs=next_ind, x=x, reuse=True)

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def init_lstm(self):
        features = self.features
        c, h = self._get_initial_lstm(features=features)
        return c, h

    def build_sampler(self, max_len=7):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        mask = tf.fill([tf.shape(features)[0], self.V], 0.0)
        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start),
                                         x=tf.zeros([self.V], tf.float32))
            else:
                x = self._word_embedding(inputs=sampled_word, x=x, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            logits += mask
            sampled_word = tf.argmax(logits, 1)
            mask += tf.to_float(tf.one_hot(sampled_word, self.V, on_value = -100))
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions

    def init_sampler(self):
        features = self.features
        features = self._batch_norm(features, mode='test', name='conv_features')
        features_proj = self._project_features(features=features, reuse=False)
        c, h = self._get_initial_lstm(features=features)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start), x=tf.zeros([self.V], tf.float32))
        context, alpha = self._attention_layer(features, features_proj, h)
        # alpha_list.append(alpha)

        if self.selector:
            context, beta = self._selector(context, h)
        #    beta_list.append(beta)

        with tf.variable_scope('lstm'):
            _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

        logits = self._decode_lstm(x, h, context)
        sampled_word = tf.argmax(logits, 1)
        # sampled_word_list.append(sampled_word)

        # alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     # (N, T, L)
        # betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        # sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)
        # return alphas, betas, sampled_captions
        return logits, c, h, alpha, x

    def word_sampler(self):
        features = self.features
        features = self._batch_norm(features, mode='test', name='conv_features', reuse=True)
        features_proj = self._project_features(features=features, reuse=True)
        c = self.c
        h = self.h
        sampled_word = self.samp
        x = self.x
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        x = self._word_embedding(inputs=sampled_word, x=x, reuse=True)
        context, alpha = self._attention_layer(features, features_proj, h, reuse=True)
        # alpha_list.append(alpha)

        if self.selector:
            context, beta = self._selector(context, h, reuse=True)
        #    beta_list.append(beta)

        with tf.variable_scope('lstm', reuse=True):
            _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

        logits = self._decode_lstm(x, h, context, reuse=True)
        sampled_word = tf.argmax(logits, 1)
        # sampled_word_list.append(sampled_word)

        # alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     # (N, T, L)
        # betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        # sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)
        # return alphas, betas, sampled_captions
        return logits, c, h, alpha, x
