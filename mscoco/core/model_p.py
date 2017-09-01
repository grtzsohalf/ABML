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
    def __init__(self, word_to_idx, idx_to_word, dim_feature=[196, 1024], dim_embed=512, dim_hidden=1024, \
                 n_time_step=16, prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, \
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
        self.idx_to_word = idx_to_word
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
        self.init_pred = tf.placeholder(tf.float32, [None, self.V - 3])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.groundtruth = tf.placeholder(tf.float32, [None, self.V])
        self.c = tf.placeholder(tf.float32, [1,1024])
        self.h = tf.placeholder(tf.float32, [1,1024])
        self.samp = tf.placeholder(tf.int32, [1])
        # self.y = tf.placeholder(tf.float32, [1, self.V -3])
        self.p = tf.placeholder(tf.float32, [1, self.V -3])

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

    def _word_embedding(self, inputs, p, reuse=False):
        '''
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [batch_size, batch_size*self.M], initializer=self.emb_initializer)
            b = tf.get_variable('b', [batch_size*self.M], initializer=self.const_initializer)
            x = tf.reshape(tf.nn.sigmoid(tf.matmul(inputs, w) + b, 'x'), [batch_size, self.M])
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
        '''
        p += tf.to_float(tf.one_hot(inputs, self.V-3, on_value=1))
        return p

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

    def _decode_lstm(self, h, context, p, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.V], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.V], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.V + (self.V-3), self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.8)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.V], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            h_logits = tf.nn.tanh(h_logits)
            if self.prev2out:
                h_logits = tf.concat(1, [h_logits, p])

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.8)
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
        batch_size = tf.shape(features)[0]
        groundtruth = tf.to_float(self.groundtruth[:, 3:])
        predicted = tf.zeros([batch_size, self.V-3], tf.float32)

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        # y = init_pred
        features_proj = self._project_features(features=features)
        p = tf.zeros([batch_size, self.V-3], tf.float32)
        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(5):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [context, p]), state=[c, h])

            logits = self._decode_lstm(h, context, p, dropout=self.dropout, reuse=(t!=0))
            logits = logits[:, 3:]
            # logits = tf.Print(logits, [logits], message="logits = ", summarize=10)
            loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, groundtruth))

            unpredicted_labels = logits + predicted
            next_ind = tf.argmax(unpredicted_labels, 1)
            predicted += tf.to_float(tf.one_hot(next_ind, self.V-3, on_value=-100))
            p = self._word_embedding(inputs=next_ind, p=p, reuse=True)

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

    def build_sampler(self, max_len=15):
        features = self.features
        init_pred = self.init_pred

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
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [context, init_pred]), state=[c, h])

            logits = self._decode_lstm(h, context, init_pred, reuse=(t!=0))
            init_pred = logits[:, 3:]
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
        # y = self.init_pred
        p = tf.zeros([tf.shape(features)[0], self.V-3], tf.float32)
        features = self._batch_norm(features, mode='test', name='conv_features')
        features_proj = self._project_features(features=features, reuse=False)
        c, h = self._get_initial_lstm(features=features)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        context, alpha = self._attention_layer(features, features_proj, h)

        if self.selector:
            context, beta = self._selector(context, h)

        with tf.variable_scope('lstm'):
            _, (c, h) = lstm_cell(inputs=tf.concat(1, [context, p]), state=[c, h])

        logits = self._decode_lstm(h, context, p)
        return logits, c, h, alpha, p

    def word_sampler(self):
        features = self.features
        features = self._batch_norm(features, mode='test', name='conv_features', reuse=True)
        features_proj = self._project_features(features=features, reuse=True)
        c = self.c
        h = self.h
        sampled_word = self.samp
        p = self.p
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        p = self._word_embedding(inputs=sampled_word, p=p, reuse=True)
        context, alpha = self._attention_layer(features, features_proj, h, reuse=True)
        # alpha_list.append(alpha)

        if self.selector:
            context, beta = self._selector(context, h, reuse=True)

        with tf.variable_scope('lstm', reuse=True):
            _, (c, h) = lstm_cell(inputs=tf.concat(1, [context, p]), state=[c, h])

        logits = self._decode_lstm(h, context, p, reuse=True)
        return logits, c, h, alpha, p
