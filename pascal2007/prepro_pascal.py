from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils_pascal import *

import tensorflow as tf
import numpy as np
import pandas as pd
from random import *
import hickle
import os
import json
os.environ['CUDA_VISIBLE_DEVICES']='0'

def _process_caption_data(caption_file, image_dir, max_length=7):
    with open(caption_file) as f:
        caption_data = json.load(f)
    # build word to idx
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    dic = {}
    idx_to_word = {}
    idx_counter = {}
    for cate in caption_data['categories']:
        idx_counter[cate['name']] = 0
        idx_to_word[cate['id']] = cate['name']
        dic[cate['name']] = 0
    idx = 3
    for key in sorted(dic.iterkeys()):
        word_to_idx[key] = idx
        idx += 1
    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}
    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = {}
    annotation_num = 0
    label_num = 0
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        idx_counter[idx_to_word[annotation['category_id']]] += 1
        if not image_id in data:
            annotation_num += 1
            data[image_id] = {}
            data[image_id]['caption'] = []
            data[image_id]['image_id'] = image_id
            annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
            data[image_id]['file_name'] = annotation['file_name']
        category_name = idx_to_word[annotation['category_id']]
        caption_index = word_to_idx[category_name]
        if not caption_index in data[image_id]['caption']:
            data[image_id]['caption'] += [caption_index]
    for k, v in idx_counter.iteritems():
        print k, v
    max_len = 0
    for image in data:
        caption_len = len(data[image]['caption'])
        if caption_len > max_len:
            max_len = caption_len
    print 'Max length of caption: ', max_len
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data, orient='index')
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        if len(caption) > max_length:
            del_idx.append(i)
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    print "The number of captions after deletion: %d" %len(caption_data)
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    print "The number of captions: %d" %len(caption_data)
    return caption_data, word_to_idx


def _build_vocab(caption_file, threshold=1):
    with open(caption_file) as f:
        caption_data = json.load(f)
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    print "categories", caption_data['categories']
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=7):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32) 

    for i, caption in enumerate(annotations['caption']):
        # words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for label in caption:
            cap_vec.append(label)
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 7
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path
    # vgg_model_path = '/home/jason6582/sfyc/attention-tensorflow/imagenet-vgg-verydeep-19.mat'

    train_caption_file = '/home/jason6582/sfyc/pascal_json/pascal_train2007.json'
    val_caption_file = '/home/jason6582/sfyc/pascal_json/pascal_val2007.json'
    test_caption_file = '/home/jason6582/sfyc/pascal_json/pascal_test2007.json'
    train_dataset, word_to_idx = _process_caption_data(caption_file=train_caption_file,
                    image_dir='/home/jason6582/sfyc/voc_train/resized_images/', max_length=7)
    val_dataset, word_to_idx = _process_caption_data(caption_file=val_caption_file,
                    image_dir='/home/jason6582/sfyc/voc_train/resized_images/', max_length=7)
    test_dataset, _ = _process_caption_data(caption_file=test_caption_file,
                    image_dir='/home/jason6582/sfyc/voc_test/resized_images/', max_length=7)

    print 'Finished processing caption data'
    train_dataset = pd.concat([train_dataset, val_dataset[250:]]).reset_index(drop=True)
    val_dataset = val_dataset[:250]
    print len(train_dataset)
    print len(val_dataset)

    save_pickle(train_dataset, 'pascaldata/train/train.annotations.pkl')
    save_pickle(val_dataset, 'pascaldata/val/val.annotations.pkl')
    save_pickle(test_dataset.reset_index(drop=True), 'pascaldata/test/test.annotations.pkl')
    save_pickle(word_to_idx, 'pascaldata/train/word_to_idx.pkl')

    for split in ['train', 'val', 'test']:
        if split == 'train':
            max_length = 7
        else:
            max_length = 20
        annotations = load_pickle('./pascaldata/%s/%s.annotations.pkl' % (split, split))
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './pascaldata/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './pascaldata/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './pascaldata/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption)
        save_pickle(feature_to_captions, './pascaldata/%s/%s.references.pkl' % (split, split))
        print "Finished building %s caption dataset" %split

    # extract conv5_3 feature vectors
    '''
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'val', 'test']:
            anno_path = './pascaldata/%s/%s.annotations.pkl' % (split, split)
            save_path = './pascaldata/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)
            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
            # print all_feats.shape
            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                        np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))
                # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))
    '''

if __name__ == "__main__":
    main()
