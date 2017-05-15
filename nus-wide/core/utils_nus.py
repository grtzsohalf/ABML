import numpy as np
import cPickle as pickle
import hickle
import time
import os

def load_word_to_idx(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    if split == 'train':
        with open(os.path.join(data_path, 'word_to_idx81.pkl'), 'rb') as f:
            word = pickle.load(f)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return word

def load_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    data['features'] = hickle.load(os.path.join(data_path, '%s.features81.hkl' % (split)))
    with open(os.path.join(data_path, '%s.file.names81.pkl' % (split)), 'rb') as f:
        data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.captions81.pkl' % (split)), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs81.pkl' % (split)), 'rb') as f:
        data['image_idxs'] = pickle.load(f)
    '''
    n_examples = len(data['image_idxs'])
    rand_idxs = np.random.permutation(n_examples)
    data['file_names'] = data['file_names'][rand_idxs]
    data['captions'] = data['captions'][rand_idxs]
    data['image_idxs'] = data['image_idxs'][rand_idxs]
    data['features'] = data['features'][rand_idxs]
    train_cutoff = [0]
    for i in range(p-1):
        train_cutoff.append(int(n_examples/16)*(i+1))
    for part in range(p-1):
        hickle.dump(data['features'][train_cutoff[part]:train_cutoff[part+1]], os.path.join(data_path, '%s.features81_%s.hkl' % (split, str(part))))
        save_pickle(data['file_names'][train_cutoff[part]:train_cutoff[part+1]], os.path.join(data_path, '%s.file.names81_%s.pkl' % (split, str(part))))
        save_pickle(data['captions'][train_cutoff[part]:train_cutoff[part+1]], os.path.join(data_path, '%s.captions81_%s.pkl' % (split, str(part))))
        save_pickle(data['image_idxs'][train_cutoff[part]:train_cutoff[part+1]], os.path.join(data_path, '%s.image_idxs81_%s.pkl' % (split, str(part))))
    hickle.dump(data['features'][train_cutoff[p-1]:], os.path.join(data_path, '%s.features81_%s.hkl' % (split, str(p-1))))
    save_pickle(data['file_names'][train_cutoff[p-1]:], os.path.join(data_path, '%s.file.names81_%s.pkl' % (split, str(p-1))))
    save_pickle(data['captions'][train_cutoff[p-1]:], os.path.join(data_path, '%s.captions81_%s.pkl' % (split, str(p-1))))
    save_pickle(data['image_idxs'][train_cutoff[p-1]:], os.path.join(data_path, '%s.image_idxs81_%s.pkl' % (split, str(p-1))))
    '''
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return data

def load_coco_data(data_path='./data', split='train', part=''):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    if split == 'train':
        data['features'] = hickle.load(os.path.join(data_path, '%s.features81_%s.hkl' % (split, part)))
        with open(os.path.join(data_path, '%s.file.names81_%s.pkl' % (split, part)), 'rb') as f:
            data['file_names'] = pickle.load(f)
        with open(os.path.join(data_path, '%s.captions81_%s.pkl' % (split, part)), 'rb') as f:
            data['captions'] = pickle.load(f)
        with open(os.path.join(data_path, '%s.image.idxs81_%s.pkl' % (split, part)), 'rb') as f:
            data['image_idxs'] = pickle.load(f)
        with open(os.path.join(data_path, 'word_to_idx81.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)
        '''
        for k, v in data.iteritems():
            if type(v) == np.ndarray:
                print k, type(v), v.shape, v.dtype
            else:
                print k, type(v), len(v)
        '''
    else:
        data['features'] = hickle.load(os.path.join(data_path, '%s.features81.hkl' % split))
        with open(os.path.join(data_path, '%s.file.names81.pkl' %split), 'rb') as f:
            data['file_names'] = pickle.load(f)
        with open(os.path.join(data_path, '%s.captions81.pkl' %split), 'rb') as f:
            data['captions'] = pickle.load(f)
        with open(os.path.join(data_path, '%s.image.idxs81.pkl' %split), 'rb') as f:
            data['image_idxs'] = pickle.load(f)
        for k, v in data.iteritems():
            if type(v) == np.ndarray:
                print k, type(v), v.shape, v.dtype
            else:
                print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return data

def decode_captions(captions, idx_to_word):
    # for i in idx_to_word.iteritems():
    #     print i
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            # if word == '<END>':
            #   words.append('END')
            #   break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def decode_py_captions(captions, idx_to_word):
    N = len(captions)
    decoded = []
    for i in range(N):
        words = []
        T = len(captions[i])
        for t in range(T):
            word = idx_to_word[captions[i][t] + 3]
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names

def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('METEOR: %f\n' %scores['METEOR'])  
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)
