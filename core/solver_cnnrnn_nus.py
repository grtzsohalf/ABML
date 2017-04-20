import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data_path, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data_path = data_path
        self.data = None
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.V = kwargs.pop('V', 84)
        self.n_time_step = kwargs.pop('n_time_step', 11)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer   

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        generated_captions = self.model.build_sampler(max_len=10)

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
           
        # summary op   
        tf.scalar_summary('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.histogram_summary(var.op.name+'/gradient', grad)
        
        summary_op = tf.merge_all_summaries() 

        print "The number of epoch: %d" %self.n_epochs
        print "Batch size: %d" %self.batch_size
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=100)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)
            prev_loss = []
            curr_loss = []
            for i in range(50):
                prev_loss.append(-1)
                curr_loss.append(0)
            start_t = time.time()
            for e in range(self.n_epochs):
                arr = np.arange(50)
                np.random.shuffle(arr)
                for part in range(50):
                    p = arr[part]
                    print '##################'
                    print 'part ' + str(p+1) + ' of ' + 'epoch ' + str(e+1)
                    print '##################'
                    self.data = load_coco_data(data_path=self.data_path, split='train', part=str(p))
                    n_examples = self.data['captions'].shape[0]
                    n_iters_per_part = int(np.ceil(float(n_examples)/self.batch_size))
                    features = self.data['features']
                    captions = self.data['captions']

                    image_idxs = self.data['image_idxs']
                    print "Data size: %d" %n_examples
                    print "Iterations per part: %d" %n_iters_per_part

                    rand_idxs = np.random.permutation(n_examples)
                    captions = captions[rand_idxs]
                    image_idxs = image_idxs[rand_idxs]

                    for i in range(n_iters_per_part):
                        '''
                        if i == n_iters_per_part - 1:
                            continue
                        '''
                        captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                        image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                        features_batch = features[image_idxs_batch]

                        feed_dict = {self.model.features: features_batch, 
                                     self.model.captions: captions_batch} 
                        _, l = sess.run([train_op, loss], feed_dict)
                        curr_loss[p] += l

                        # write summary for tensorboard visualization
                        if i % 10 == 0:
                            summary = sess.run(summary_op, feed_dict)
                            summary_writer.add_summary(summary, e*n_iters_per_part + i)
                        if (i+1) % self.print_every == 0:
                            # print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                            ground_truths = captions[image_idxs == image_idxs_batch[0]]
                            decoded = decode_captions(ground_truths, self.model.idx_to_word)
                            for j, gt in enumerate(decoded):
                                print "Ground truth %d: %s" %(j+1, gt)
                            gen_caps = sess.run(generated_captions, feed_dict)
                            decoded = decode_captions(gen_caps, self.model.idx_to_word)
                            print "Generated caption: %s\n" %decoded[0]

                    print "Previous epoch loss (part %s): " % str(p+1), prev_loss[p]
                    print "Current epoch loss (part %s): " % str(p+1), curr_loss[p]
                    print "Elapsed time: ", time.time() - start_t
                    prev_loss[p] = curr_loss[p]
                    curr_loss[p] = 0
                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'nus_cnnrnn'), global_step=e+1)
                    print "nus_cnnrnn-%s saved." %(e+1)

    def softmax(self, array):
        total = 0.0
        for i in range(len(array)):
            array[i] = np.exp(array[i])
            total += array[i]
        return array/total

    def sigmoid(self, array):
        for i in range(len(array)):
            array[i] = 1/(1 + np.exp(-array[i]))
        return array

    def bpmll(self, array):
        for i in range(len(array)):
            array[i] = np.exp(array[i]-1)
        return array

    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True,\
             filename='', thres=0.0):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        print self.model.idx_to_word

        features = data['features']

        # build a graph to sample captions

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        probabilities_start, c_start, h_start, x_start = self.model.init_sampler()
        probabilities, c, h, x = self.model.word_sampler()
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            # feed_dict = { self.model.features: features_batch }
            MIN_LEN = 3
            MAX_LEN = 10
            K = 3 # beam search width
            # all_sam_cap = np.zeros((features.shape[0], MAX_LEN), dtype = int)
            num_iter = features.shape[0]
            start_t = time.time()
            for thres_iter in range(1):
                all_sam_cap = []
                # THRES = 0.05*(thres_iter+1)
                for i in range(num_iter):
                    if i % 50 == 0:
                        print "Iteration: ", i
                    features_batch = features[i:i+1]
                    pathProbs = []
                    for k in range(K):
                        pathProbs.append(1.0)
                    x_run = None

                    candidate_paths = {}
                    THRES = 0.0
                    for t in range(MAX_LEN):
                        dic = {}
                        for j in range(K):
                            if t == 0:
                                path = []
                                feed_dict = { self.model.features: features_batch }
                                probsNumpy, c_run, h_run, x_run = \
                                sess.run([probabilities_start, c_start, h_start, \
                                          x_start], feed_dict)
                                probsNumpy = probsNumpy.reshape(84)
                            else:
                                if j >= len(paths_info):
                                    break
                                path, c_run, h_run, samp_run, x_run = paths_info[j]
                                feed_dict = { self.model.features: features_batch,
                                                self.model.c: c_run,
                                                self.model.h: h_run,
                                                self.model.samp: samp_run,
                                                self.model.x: x_run }
                                probsNumpy, c_run, h_run, x_run = \
                                sess.run([probabilities, c, h, x], feed_dict)
                                probsNumpy = probsNumpy.reshape(84)
                            probsNumpy = self.softmax(probsNumpy)
                            probs = []
                            for p in probsNumpy:
                                probs.append(p)
                            if len(path) != 0:
                                for predicted in path:
                                    probs[predicted] = 0.0
                                if t < MIN_LEN:
                                    probs[0] = 0.0
                            for k in range(len(probs)):
                                idx = probs[k] * pathProbs[j]
                                p = path[:]
                                p.append(k)
                                samp_run = np.array([k])
                                dic[idx] = (p, c_run , h_run, samp_run, x_run) # p is a path(list), and h is p's current hidden state
                        count = 0
                        newPaths_info = []
                        newPathProbs = []
                        for key in reversed(sorted(dic.iterkeys())):
                            count += 1
                            if count > K:
                                break
                            if dic[key][0][-1] == 0:
                                candidate_paths[key] = dic[key][:]
                                continue
                            newPaths_info.append(dic[key][:])
                            newPathProbs.append(key)
                        if len(candidate_paths) != 0:
                            THRES = list(reversed(sorted(candidate_paths.iterkeys())))[0]
                        if len(newPathProbs) == 0:
                            break
                        if t != 0 and newPathProbs[0] < THRES:
                            break
                        paths_info = newPaths_info
                        pathProbs = newPathProbs
                    all_sam_cap.append(candidate_paths[THRES][0][:-1])
                all_decoded = decode_by_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./nusdata/%s/%s.candidate.captions81_%s.pkl" % \
                            (split, split, filename))
