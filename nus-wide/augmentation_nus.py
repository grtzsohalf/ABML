from PIL import Image
import os
import json
import numpy as np
from core.utils import *
from random import *
import time
import math

def process_data():
    tagListFile = '/home/jason6582/sfyc/NUS-WIDE/tags/Concepts81.txt'
    imageListFile1 = '/home/jason6582/sfyc/NUS-WIDE/ImageList/TrainImagelist.txt'
    tagsFile1 = '/home/jason6582/sfyc/NUS-WIDE/tags/label1.txt'
    imageListFile2 = '/home/jason6582/sfyc/NUS-WIDE/ImageList/TestImagelist.txt'
    tagsFile2 = '/home/jason6582/sfyc/NUS-WIDE/tags/label2.txt'

    tags = []
    with open(tagListFile, 'r') as f:
        for line in f:
            tags.append(line[:-2])
    # tags = sorted(tags, key = str.lower)

    annotations = []
    availableID = []
    count = 0
    with open(tagsFile1, 'r') as f:
        for line in f:
            caption = ''
            labels = line.split()
            for i in range(len(labels)):
                if labels[i][:1] == '1':
                    caption += (tags[i] + ' ')
            if caption != '':
                caption = caption[:-1]
                # caption += 'END'
                annotations.append({'image_id':count, 'id':count, 'caption':caption})
                availableID.append(1)
            else:
                availableID.append(0)
            count += 1
    with open(tagsFile2, 'r') as f:
        for line in f:
            caption = ''
            labels = line.split()
            for i in range(len(labels)):
                if labels[i][:1] == '1':
                    caption += (tags[i] + ' ')
            if caption != '':
                caption = caption[:-1]
                # caption += 'END'
                annotations.append({'image_id':count, 'id':count, 'caption':caption})
                availableID.append(1)
            else:
                availableID.append(0)
            count += 1
        print count
        print len(annotations)

    count = 0
    available = 0
    with open(imageListFile1, 'r') as f:
        for line in f:
            if availableID[count] == 1:
                annotations[available]['file_name'] = line[:-1]
                available += 1
            count += 1
    with open(imageListFile2, 'r') as f:
        for line in f:
            if availableID[count] == 1:
                annotations[available]['file_name'] = line[:-1]
                available += 1
            count += 1

    shuffle(annotations)
    for a in annotations[:10]:
        print a

    return annotations

def crop(image):
    image = image.resize([256, 256], Image.ANTIALIAS)
    width = 256
    height = 256
    arr = [168, 192, 224, 256]
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
    return [img.resize([224, 224], Image.ANTIALIAS) for img in image_list]

def main():
    # previous_t = time.time()
    folder = '/home/jason6582/caffe/tool123/flickrfeature/'
    resized_folder = '/home/jason6582/sfyc/NUS-WIDE/flickrfeature_aug/'
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    data = process_data()
    train_cutoff = 142500
    '''
    count = 0
    split = 'train'
    annotations = []
    images = []
    print 'Start resizing %s images.' %split
    for i, annotation in enumerate(data[: train_cutoff]):
        image_id = annotation['image_id']
        id_string = str(image_id)
        file_name = annotation['file_name']
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                crop_images = crop(image)
                for img in crop_images:
                    caption = annotation['caption']
                    annotations.append({'image_id': count, 'id': count, 'caption': caption})
                    images.append({'id': count, 'file_name': str(count) + '.jpg'})
                    img.save(os.path.join(resized_folder, str(count)+'.jpg'), img.format)
                    count += 1
                    if count % 100 == 0:
                        # print time.time() - previous_t
                        # previous_t = time.time()
                        print 'Resized %s image: ' %split, count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/NUS-WIDE/%s_aug.json' %split, 'w') as f:
        json.dump(caption_data, f)
    '''
    count = train_cutoff * 5
    split = 'val'
    annotations = []
    images = []
    print 'Start resizing %s images.' %split
    for i, annotation in enumerate(data[train_cutoff: len(data)]):
        image_id = annotation['image_id']
        id_string = str(image_id)
        file_name = annotation['file_name']
        image_file = folder + file_name
        with open(image_file, 'r+b') as f:
            with Image.open(f) as image:
                image = image.resize([224, 224], Image.ANTIALIAS)
                # image = image.crop((16, 16, 240, 240))
                caption = annotation['caption']
                annotations.append({'image_id': count, 'id': count, 'caption': caption})
                images.append({'id': count, 'file_name': str(count) + '.jpg'})
                image.save(os.path.join(resized_folder, str(count)+'.jpg'), image.format)
                count += 1
                if count % 100 == 0:
                    # print time.time() - previous_t
                    # previous_t = time.time()
                    print 'Resized %s image: ' %split, count
    caption_data = {'images': images, 'annotations': annotations}
    with open('/home/jason6582/sfyc/NUS-WIDE/%s_aug.json' %split, 'w') as f:
        json.dump(caption_data, f)

if __name__ == "__main__":
    main()
