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
    idx = 0
    for key in sorted(dic.iterkeys()):
        word_to_idx[key] = idx
        idx += 1
    data = {}
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        if not image_id in data:
            data[image_id] = []
        category_name = idx_to_word[annotation['category_id']]
        caption_index = word_to_idx[category_name]
        data[image_id] += [(caption_index, bbox)]
    return data, word_to_idx

def union_ratio(bbox, axes):
    union_region = 0
    for x in range(int(math.floor(bbox[0])), int(math.floor(bbox[0]+bbox[2]))):
        for y in range(int(math.floor(bbox[1])), int(math.floor(bbox[1]+bbox[3]))):
            if x > axes[0] and x < axes[2] and y > axes[1] and y < axes[3]:
                union_region += 1
    return (float(union_region) / float((bbox[2])*(bbox[3]))) > 0.5

def crop(image, pair_list, split):
    o_width, o_height = image.size
    image = image.resize([256, 256], Image.ANTIALIAS)
    width, height = image.size
    ratio_x = float(width) / float(o_width)
    ratio_y = float(height) / float(o_height)
    # left, top, right, buttom
    arr = [128, 168, 192, 224]
    ran_arr = []
    for i in range(5):
        ran_arr += [np.random.choice(arr)]
    axes = [[0, 0, ran_arr[0], ran_arr[0]],
            [width-ran_arr[1], 0, width, ran_arr[1]],
            [0, height-ran_arr[2], ran_arr[2], height],
            [width-ran_arr[3], height-ran_arr[3], width, height],
            [128-int(ran_arr[4]/2), 128-int(ran_arr[4]/2),\
             128+int(ran_arr[4]/2), 128+int(ran_arr[4]/2)]  ]
    image_list = [image.crop((axes[i][0], axes[i][1], axes[i][2], axes[i][3])) for i in range(5)]
    tuple_list = []
    # t = time.time()
    for i, img in enumerate(image_list):
        label_list = []
        for label in pair_list:
            index, bbox = label
            bbox = [bbox[0]*ratio_x, bbox[1]*ratio_y, bbox[2]*ratio_x, bbox[3]*ratio_y]
            if index not in label_list:
                # if split == 'train':
                if split == 'train':
                    if union_ratio(bbox, axes[i]) == True:
                        label_list.append(index)
                else:
                    label_list.append(index)
        tuple_list.append((img.resize([224, 224], Image.ANTIALIAS), label_list))
    # print time.time() - t
    return tuple_list

def main():
    # previous_t = time.time()
    '''
    split = 'train'
    caption_file = '/home/jason6582/sfyc/mscoco/annotations/instances_%s2014.json' %split
    folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014/' %split
    resized_folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014_aug/' %split
    data, word_to_idx = process_data(caption_file = caption_file, image_dir=folder)
    cutoff = len(data) * 19 / 20
    keys = list(data.keys())
    shuffle(keys)
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    save_pickle(word_to_idx, './cocodata/%s/word2idx.pkl' % (split))
    print 'Start resizing %s images.' %split
    idx_to_word = {}
    for key in word_to_idx:
        idx_to_word[word_to_idx[key]] = key
    annotations = []
    images = []
    count = 0
    for i, image_id in enumerate(keys[:cutoff]):
        id_string = str(image_id)
        file_name = 'COCO_' + split + '2014_' + \
                        '0'*(12-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_tuples = crop(image, data[image_id], 'train')
                for t in crop_tuples:
                    caption = ''
                    for l in t[1]:
                        caption = caption + str(l) + ' '
                    if caption != '':
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        t[0].save(os.path.join(resized_folder, str(count)+'.jpg'), t[0].format)
                    count += 1
                    if count % 500 == 0:
                        print 'Resized %s image: ' %split, count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/mscoco/annotations/train_aug.json', 'w') as f:
        json.dump(caption_data, f)

    print 'Start resizing val images.'
    annotations = []
    images = []
    for i, image_id in enumerate(keys[cutoff:]):
        id_string = str(image_id)
        file_name = 'COCO_' + split + '2014_' + \
                        '0'*(12-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_tuples = crop(image, data[image_id], 'val')
                for t in crop_tuples:
                    caption = ''
                    for l in t[1]:
                        caption = caption + str(l) + ' '
                    if caption != '':
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        t[0].save(os.path.join(resized_folder, str(count)+'.jpg'), t[0].format)
                    count += 1
                    if count % 500 == 0:
                        print 'Resized val image: ', count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/mscoco/annotations/val_aug.json', 'w') as f:
        json.dump(caption_data, f)
    '''
    '''
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                image = image.resize([224, 224], Image.ANTIALIAS)
                # image = image.crop((16, 16, 240, 240))
                image.save(os.path.join(resized_folder, str(count)+'.jpg'), image.format)
        caption = ''
        caption_idx = []
        for label in data[image_id]:
            index, _ = label
            if index not in caption_idx:
                caption_idx += [index]
        for index in caption_idx:
            caption = caption + str(index) + ' '
        annotations.append({'image_id': count, 'id': count, 'caption': caption})
        images.append({'id': count, 'file_name': str(count) + '.jpg'})
        count += 1
        if count % 100 == 0:
            print 'Resized %s (val) image: ' %split, count
    '''
    '''
    split = 'val'
    caption_file = '/home/jason6582/sfyc/mscoco/annotations/instances_%s2014.json' %split
    folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014/' %split
    resized_folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014_aug/' %split
    val_data, word_to_idx = process_data(caption_file = caption_file, image_dir=folder)
    # val_data.update(dict(data.items()[cutoff:]))
    # val_data.update({key: data[key] for key in keys[cutoff:]})
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    print 'Start resizing test images.'
    annotations = []
    images = []
    count = 0
    for i, image_id in enumerate(val_data):
        id_string = str(image_id)
        file_name = 'COCO_' + split + '2014_' + \
                        '0'*(12-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_tuples = crop(image, val_data[image_id], split)
                for t in crop_tuples:
                    caption = ''
                    for l in t[1]:
                        caption = caption + str(l) + ' '
                    if caption != '':
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        t[0].save(os.path.join(resized_folder, str(count)+'.jpg'), t[0].format)
                    count += 1
                    if count % 500 == 0:
                        print 'Resized test image: ', count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/mscoco/annotations/test_aug.json', 'w') as f:
        json.dump(caption_data, f)
    '''
    '''
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                image = image.resize([224, 224], Image.ANTIALIAS)
                # image = image.crop((16, 16, 240, 240))
                image.save(os.path.join(resized_folder, str(count)+'.jpg'), image.format)
        caption = ''
        caption_idx = []
        for label in val_data[image_id]:
            index, _ = label
            if index not in caption_idx:
                caption_idx += [index]
        for index in caption_idx:
            caption = caption + str(index) + ' '
        annotations.append({'image_id': count, 'id': count, 'caption': caption})
        images.append({'id': count, 'file_name': str(count) + '.jpg'})
        count += 1
        if count % 100 == 0:
            print 'Resized %s image: ' %split, count
    '''
    '''
    ### val sampled from test
    split = 'val'
    caption_file = '/home/jason6582/sfyc/mscoco/annotations/instances_%s2014.json' %split
    folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014/' %split
    resized_folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/testval2014_aug/'
    val_data, word_to_idx = process_data(caption_file = caption_file, image_dir=folder)
    # val_data.update(dict(data.items()[cutoff:]))
    # val_data.update({key: data[key] for key in keys[cutoff:]})
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    print 'Start resizing val images.'
    annotations = []
    images = []
    count = 0
    keys = list(val_data.keys())
    rand_samp = [ keys[i] for i in sample(xrange(len(keys)), 2000) ]
    for i, image_id in enumerate(rand_samp):
        id_string = str(image_id)
        file_name = 'COCO_' + split + '2014_' + \
                        '0'*(12-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_tuples = crop(image, val_data[image_id], 'val')
                for t in crop_tuples:
                    caption = ''
                    for l in t[1]:
                        caption = caption + str(l) + ' '
                    if caption != '':
                        annotations.append({'image_id': count, 'id': count, 'caption': caption})
                        images.append({'id': count, 'file_name': str(count) + '.jpg'})
                        t[0].save(os.path.join(resized_folder, str(count)+'.jpg'), t[0].format)
                    count += 1
                    if count % 500 == 0:
                        print 'Resized val image: ', count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/mscoco/annotations/testval_aug.json', 'w') as f:
        json.dump(caption_data, f)
    '''
    split = 'val'
    caption_file = '/home/jason6582/sfyc/mscoco/annotations/instances_%s2014.json' %split
    folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/%s2014/' %split
    resized_folder = '/home/jason6582/sfyc/attention-tensorflow/mscoco/image/testval2014_aug/'
    val_data, word_to_idx = process_data(caption_file = caption_file, image_dir=folder)
    # val_data.update(dict(data.items()[cutoff:]))
    # val_data.update({key: data[key] for key in keys[cutoff:]})
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    print 'Start resizing val images.'
    annotations = []
    images = []
    count = 0
    keys = list(val_data.keys())
    rand_samp = [ keys[i] for i in sample(xrange(len(keys)), 2000) ]
    for i, image_id in enumerate(rand_samp):
        id_string = str(image_id)
        file_name = 'COCO_' + split + '2014_' + \
                        '0'*(12-len(id_string)) + id_string +'.jpg'
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                image = image.resize([224, 224], Image.ANTIALIAS)
                # image = image.crop((16, 16, 240, 240))
                image.save(os.path.join(resized_folder, str(count)+'.jpg'), image.format)
        caption = ''
        caption_idx = []
        for label in val_data[image_id]:
            index, _ = label
            if index not in caption_idx:
                caption_idx += [index]
        for index in caption_idx:
            caption = caption + str(index) + ' '
        annotations.append({'image_id': count, 'id': count, 'caption': caption})
        images.append({'id': count, 'file_name': str(count) + '.jpg'})
        count += 1
        if count % 100 == 0:
            print 'Resized %s image: ' %split, count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/mscoco/annotations/testval_aug.json', 'w') as f:
        json.dump(caption_data, f)

if __name__ == '__main__':
    main()
