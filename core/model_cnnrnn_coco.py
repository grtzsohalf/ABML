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


class CaptionGenerator(object):
    def __init__(self, word_to_idx, idx_to_word, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=11, 
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        
        self.word_to_idx = word_to_idx
        #self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
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

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.c = tf.placeholder(tf.float32, [1,self.H])
        self.h = tf.placeholder(tf.float32, [1,self.H])
        self.samp = tf.placeholder(tf.int32, [1])
        self.x = tf.placeholder(tf.float32, [1,self.M])
   
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

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
    '''
    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj
    '''
    def _decode_lstm(self, h, features, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            w_feat = tf.get_variable('w_feat', [self.L*self.D, self.M], initializer=self.weight_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)

            features = tf.reshape(features, [-1, self.L*self.D])
            
            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h)

            if dropout:
                features = tf.nn.dropout(features, 0.5)
            feat_logits = tf.matmul(features, w_feat)

            out_logits = tf.nn.relu(tf.matmul(h_logits + feat_logits, w_out))
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

        captions_in = captions[:, :self.T]      
        captions_out = captions[:, 1:]  
        mask = tf.to_float(tf.not_equal(captions_out, self._null))
        
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')
        
        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        # features_proj = self._project_features(features=features)

        loss = 0.0
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x[:,t,:], state=[c, h])

            logits = self._decode_lstm(h, features, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=15):
        features = self.features
        
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        
        c, h = self._get_initial_lstm(features=features)
        #features_proj = self._project_features(features=features)

        sampled_word_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)  

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x, state=[c, h])

            logits = self._decode_lstm(h, features, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)       
            sampled_word_list.append(sampled_word)     

        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)
        return sampled_captions

    def init_sampler(self):
        features = self.features
        features = self._batch_norm(features, mode='test', name='conv_features')
        #features_proj = self._project_features(features=features, reuse=False)
        c, h = self._get_initial_lstm(features=features)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))

        with tf.variable_scope('lstm'):
            _, (c, h) = lstm_cell(inputs=x, state=[c, h])

        logits = self._decode_lstm(h, features)
        return logits, c, h, x

    def word_sampler(self):
        features = self.features
        features = self._batch_norm(features, mode='test', name='conv_features', reuse=True)
        #features_proj = self._project_features(features=features, reuse=True)
        c = self.c
        h = self.h
        sampled_word = self.samp
        x = self.x
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        x = self._word_embedding(inputs=sampled_word, reuse=True)  

        with tf.variable_scope('lstm', reuse=True):
            _, (c, h) = lstm_cell(inputs=x, state=[c, h])

        logits = self._decode_lstm(h, features, reuse=True)
        return logits, c, h, x
