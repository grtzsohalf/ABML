from PIL import Image
import os
import json
import numpy as np
from core.utils import *
from random import *
import time
import math

def process_data(caption_file, image_dir):
    with open(caption_file) as f:
        caption_data = json.load(f)
    # build word to idx
    dic = {}
    idx_to_word = {}
    word_to_idx = {}
    for cate in caption_data['categories']:
        idx_to_word[cate['id']] = cate['name']
        dic[cate['name']] = 0
    #print idx_to_word
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for key in sorted(dic.iterkeys()):
        word_to_idx[key] = idx
        idx += 1
    #print word_to_idx
    data = {}
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        if not image_id in data:
            data[image_id] = []
        category_name = idx_to_word[annotation['category_id']]
        caption_index = word_to_idx[category_name]
        data[image_id] += [caption_index]
    '''
    for image_id in data:
        print data[image_id]
    '''
    return data, word_to_idx

def crop(image, pair_list):
    image = image.resize([256, 256], Image.ANTIALIAS)
    width, height = image.size
    # left, top, right, buttom
    arr = [168, 192, 224, 256]
    ran_arr = []
    for i in range(10):
        ran_arr += [np.random.choice(arr)]
    axes = [[0, 0, ran_arr[0], ran_arr[1]],
            [width-ran_arr[2], 0, width, ran_arr[3]],
            [0, height-ran_arr[4], ran_arr[5], height],
            [width-ran_arr[6], height-ran_arr[7], width, height],
            [128-int(ran_arr[8]/2), 128-int(ran_arr[9]/2),\
             128+int(ran_arr[8]/2), 128+int(ran_arr[9]/2)]  ]
    image_list = [image.crop((axes[i][0], axes[i][1], axes[i][2], axes[i][3])) for i in range(5)]
    tuple_list = []
    # t = time.time()
    for i, img in enumerate(image_list):
        label_list = []
        for label in pair_list:
            index = label
            if index not in label_list:
                label_list.append(index)
        #print label_list
        tuple_list.append((img.resize([224, 224], Image.ANTIALIAS), label_list))
    # print time.time() - t
    return tuple_list

def main():
    split = 'train'
    train_caption = '/home/jason6582/sfyc/pascal_json/pascal_train2007.json'
    val_caption = '/home/jason6582/sfyc/pascal_json/pascal_val2007.json'
    folder = '/home/jason6582/sfyc/voc_train/JPEGImages/'
    resized_folder = '/home/jason6582/sfyc/voc_train/aug_image/'
    train_data, word_to_idx = process_data(caption_file = train_caption, image_dir=folder)
    val_data, _ = process_data(caption_file = val_caption, image_dir=folder)
    train_data.update(val_data)
    keys = list(train_data.keys())
    shuffle(keys)
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    # save_pickle(word_to_idx, './cocodata/%s/word2idx.pkl' % (split))
    print 'Start resizing %s images.' %split
    idx_to_word = {}
    for key in word_to_idx:
        idx_to_word[word_to_idx[key]] = key
    print idx_to_word
    annotations = []
    images = []
    count = 0
    for i, image_id in enumerate(keys):
        id_string = str(image_id)
        file_name = '0'*(6-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_tuples = crop(image, train_data[image_id])
                for t in crop_tuples:
                    caption = ''
                    for l in t[1]:
                        caption = caption + str(l) + ' '
                    if caption != '':
                        #print caption
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        t[0].save(os.path.join(resized_folder, str(count)+'.jpg'), t[0].format)
                    #print [idx_to_word[int(b)] for b in caption.split()]
                    count += 1
                    if count % 500 == 0:
                        print 'Resized %s image: ' %split, count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/pascal_json/train_aug.json', 'w') as f:
        json.dump(caption_data, f)

    caption_file = '/home/jason6582/sfyc/pascal_json/pascal_test2007.json'
    folder = '/home/jason6582/sfyc/voc_test/JPEGImages/'
    resized_folder = '/home/jason6582/sfyc/voc_test/aug_image/'
    test_data, word_to_idx = process_data(caption_file = caption_file, image_dir=folder)
    test_keys = list(test_data.keys())
    shuffle(test_keys)
    count = 0
    for split in ['val', 'test']:
        if split == 'val':
            keys = [test_keys[i] for i in sample(xrange(len(test_keys)), 500)]
        else:
            keys = test_keys
        # val_data.update(dict(data.items()[cutoff:]))
        # val_data.update({key: data[key] for key in keys[cutoff:]})
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print 'Start resizing %s images.' %split
        annotations = []
        images = []
        for i, image_id in enumerate(keys):
            id_string = str(image_id)
            file_name = '0'*(6-len(id_string)) + id_string +'.jpg'
            image_file = folder + file_name
            with open(image_file, 'r+b') as f:
                with Image.open(f) as image:
                    #crop_tuples = crop(image, test_data[image_id])
                    image = image.resize([224, 224], Image.ANTIALIAS)
                    caption = ''
                    for l in test_data[image_id]:
                        if str(l) not in caption.split():
                            caption = caption + str(l) + ' '
                    if caption != '':
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        image.save(os.path.join(resized_folder, str(count)+'.jpg'), image.format)
                    count += 1
                    if count % 500 == 0:
                        print 'Resized %s image: ' %split, count
        caption_data = {'images': images, 'annotations': annotations}
        with open('/home/jason6582/sfyc/pascal_json/%s_aug.json' %split, 'w') as f:
            json.dump(caption_data, f)

if __name__ == '__main__':
    main()
