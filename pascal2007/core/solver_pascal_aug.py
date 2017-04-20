import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import math
import os
import cPickle as pickle
from scipy import ndimage
from utils_pascal_aug import *
from bleu import evaluate
from math import *

class CaptioningSolver(object):
    def __init__(self, model, data_path, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data_path:
              Contain training data; dictionary with the following keys:
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
        self.V = kwargs.pop('V', 23)
        self.n_time_step = kwargs.pop('n_time_step', 8)

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
        '''
        # val dataset
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))
        '''
        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        _, _, generated_captions = self.model.build_sampler(max_len=7)

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
            # deprecated??
            tf.initialize_all_variables().run()
            # sess.run(tf.global_variables_initializer())
            summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=100)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)
            prev_loss = -1
            curr_loss = -1
            for e in range(self.n_epochs):
                start_t = time.time()
                print '****************'
                print 'Epoch: ', e+1
                print '****************'
                self.data = load_coco_data(data_path=self.data_path, split='train')
                n_examples = self.data['captions'].shape[0]
                n_iters_per_part = int(np.ceil(float(n_examples)/self.batch_size))
                features = self.data['features']
                captions = self.data['captions']
                '''
                for u in range(100):
                    print [self.model.idx_to_word[o] for o in captions[u]]
                '''

                # groundtruth, logits_mask and end_time
                groundtruth = np.zeros((n_examples, self.V), dtype=np.float32)
                for n, caption in enumerate(captions):
                    for index in caption:
                        if index > 0:
                            groundtruth[n][index] = 1.0

                label_num = np.sum(groundtruth, axis=1)
                masks = np.ones((self.n_time_step, n_examples, self.V), dtype=np.float32)
                for t in range(self.n_time_step):
                    for i, n in enumerate(label_num):
                        if t >= n:
                            masks[t][i] = np.zeros(self.V, dtype=np.float32)

                image_idxs = self.data['image_idxs']
                print "Data size: %d" %n_examples
                print "Iterations per part: %d" %n_iters_per_part

                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                groundtruth = groundtruth[rand_idxs]
                image_idxs = image_idxs[rand_idxs]
                masks = masks[:, rand_idxs, :]

                for i in range(n_iters_per_part):
                    '''
                    if i == n_iters_per_part - 1:
                        continue
                    '''
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    groundtruth_batch = groundtruth[i*self.batch_size:(i+1)*self.batch_size]
                    masks_batch = masks[:, i*self.batch_size:(i+1)*self.batch_size, :]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    self.model.set_batch_size(len(captions_batch))

                    # set end_time
                    '''
                    end_time = []
                    for n, caption in enumerate(captions_batch):
                        for index in caption:
                            if index == 2:
                                end_time.append(index)
                    self.model.set_end_time(end_time)
                    '''
                    feed_dict = {self.model.features: features_batch, 
                                    self.model.captions: captions_batch, 
                                    self.model.groundtruth: groundtruth_batch, 
                                    self.model.masks: masks_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

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

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0
                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'pascal_aug_ratio'), global_step=e+1)
                    print "pascal_aug_ratio-%s saved." %(e+1)

    def softmax(self, array):
        total = 0.0
        for i in range(len(array)):
            array[i] = np.exp(array[i])
            total += array[i]
        return array/total

    def sigmoid(self, array):
        array = array[3:]
        for i in range(len(array)):
            array[i] = 1/(1 + np.exp(-array[i]))
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

        features = data['features']
        reference = load_pickle('./pascaldata_aug/%s/%s.references.pkl' %(split, split))
        merge_reference = []
        for key, value in reference.iteritems():
            value = value[0].split()[:-1]
            merge_list = []
            for label in value:
                merge_list += [int(label)-3]
            merge_reference += [merge_list]
        reference = merge_reference
        ap_list = []
        for i in range(20):
            ap_list.append([])

        # build a graph to sample captions
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        probabilities_start, c_start, h_start, alpha_start, x_start = self.model.init_sampler()
        probabilities, c, h, alpha, x = self.model.word_sampler()
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            # feed_dict = { self.model.features: features_batch }
            MAX_LEN = 20
            K = 1 # beam search width
            # all_sam_cap = np.zeros((features.shape[0], MAX_LEN), dtype = int)
            num_iter = features.shape[0]
            start_t = time.time()
            for thres_iter in range(1):
                all_sam_cap = []
                all_alphas = []
                # THRES = 0.05*(thres_iter+1)
                THRES = thres
                for i in range(num_iter):
                    if i % 50 == 0:
                        print "Iteration: ", i
                    features_batch = features[i:i+1]
                    pathProbs = []
                    for k in range(K):
                        pathProbs.append(1.0)

                    x_run = None
                    history = {}
                    predicted = []
                    for t in range(MAX_LEN): # time step
                        dic = {}
                        for j in range(K):
                            if t == 0:
                                path = []
                                alphas = []
                                feed_dict = { self.model.features: features_batch }
                                probsNumpy, c_run, h_run, alpha_run, x_run = \
                                sess.run([probabilities_start, c_start, h_start, alpha_start, \
                                          x_start], feed_dict)
                                probsNumpy = probsNumpy.reshape(self.V)
                            else:
                                path, c_run, h_run, alphas, samp_run, x_run = paths_info[j]
                                feed_dict = { self.model.features: features_batch,
                                                self.model.c: c_run,
                                                self.model.h: h_run,
                                                self.model.samp: samp_run,
                                                self.model.x: x_run }
                                probsNumpy, c_run, h_run, alpha_run, x_run = \
                                sess.run([probabilities, c, h, alpha, x], feed_dict)
                                probsNumpy = probsNumpy.reshape(self.V)
                            # probsNumpy = self.softmax(probsNumpy)
                            probsNumpy = self.sigmoid(probsNumpy)
                            #print probsNumpy
                            if t == 0:
                                for q in range(len(probsNumpy)):
                                    history[q] = probsNumpy[q]
                            else:
                                for q in range(len(probsNumpy)):
                                    if q not in predicted:
                                        history[q] = probsNumpy[q] + history[q]
                            probs = [p for p in probsNumpy]
                            if len(path) != 0:
                                for predicted_idx in path:
                                    probs[predicted_idx] = 0
                            argMax = np.argmax(np.asarray(probs))
                            history[argMax] = history[argMax] /float(t+1)
                            predicted.append(argMax)
                            alphas.append(alpha_run)
                            for k in range(len(probs)):
                                idx = probs[k] * pathProbs[j]
                                p = path[:]
                                p.append(k)
                                samp_run = np.array([k+3])
                                dic[idx] = (p, c_run , h_run, alphas, samp_run, x_run) # p is a path(list), and h is p's current hidden state
                                #history[tuple(p)] = probs[k]
                        count = 0
                        newPaths_info = []
                        newPathProbs = []
                        for key in reversed(sorted(dic.iterkeys())):
                            count += 1
                            if count > K:
                                break
                            newPaths_info.append(dic[key][:])
                            # print newPaths_info[0]
                            newPathProbs.append(key)
                        if t != 0:
                            if newPathProbs[0] < THRES:
                                break
                        paths_info = newPaths_info
                        pathProbs = newPathProbs
                        #all_sam_cap.append(paths_info[0][0])
                    '''
                    for y in range(len(paths_info[0][0])):
                        sub_cap = paths_info[0][0][:y+1]
                        all_sam_cap.append((sub_cap, history[tuple(sub_cap)]))
                    '''
                    for y in range(20):
                        if y not in predicted:
                            history[y] = history[y] /float(MAX_LEN)
                        all_sam_cap.append((y, history[y]))
                    alphas = paths_info[0][3]
                    alpha_list = np.transpose(alphas, (1, 0, 2))     # (N, T, L)
                    all_alphas.append(alpha_list)
                #print all_sam_cap
                sam_cap = [t[0] for t in all_sam_cap]
                sam_prob = [t[1] for t in all_sam_cap]
                #print sam_cap
                #all_decoded = decode_py_captions([s[0] for s in all_sam_cap], self.model.idx_to_word)
                for o, sam in enumerate(sam_cap):
                    # print 'sam', sam
                    if sam in reference[int(floor(o/20))]:
                        # print reference[int(floor(o/20))]
                        ap_list[sam].append((sam_prob[o], 1))
                    else:
                        ap_list[sam].append((sam_prob[o], 0))
                save_pickle(ap_list, "./pascaldata_aug/%s/%s.candidate.captions_%s_%s.pkl" % \
                            (split, split, filename, THRES))
                print "Time cost: ", time.time()- start_t
            image_file_name = 'visualization/'
            if attention_visualization:
                # plt.rcParams['figure.figsize'] = (8.0, 6.0)
                # plt.rcParams['image.interpolation'] = 'nearest'
                # plt.rcParams['figure.cmap'] = 'gray'
                num_samples = len(all_decoded)
                ran_arr = np.random.randint(num_samples, size=10)
                reference = load_pickle('./pascaldata_aug/%s/%s.references.pkl' %(split, split))
                sample_file = open('sample.txt', 'w')
                sample_file.write('Grountruth:\n')
                for idx in ran_arr:
                    sample_file.write(str(reference[idx][0][:-2])+'\n')
                sample_file.write('\nSample:\n')
                for n in range(10):
                    k = ran_arr[n]
                    print "Sampled Caption: %s" % all_decoded[k]
                    sample_file.write(str(all_decoded[k]+'\n'))
                    # Plot original image
                    img = ndimage.imread(data['file_names'][k])
                    # plt.subplot(4, 5, 1)
                    # plt.imshow(img)
                    # plt.axis('off')
                    fname = 'origin' + str(n+1) + '.png'
                    fname = image_file_name + fname
                    plt.imsave(fname, img)
                    # Plot images with attention weights
                    words = all_decoded[k].split(" ")
                    for t in range(len(words)):
                        # plt.subplot(4, 5, t+2)
                        # plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        alp_curr = np.asarray(all_alphas[k][0][t][:]).reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        fname = 'atten_' + str(n+1) + '_' + str(t) +'.png'
                        fname = image_file_name + fname
                        plt.imsave(fname, alp_img, cmap='gray')
                        # plt.axis('off')
                    # plt.show()
            '''
            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 25))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions81_beam.pkl" %(split,split))
            '''
