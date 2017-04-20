from random import shuffle
import json

tagListFile = '/home/jason6582/sfyc/NUS-WIDE/tags/TagList1k.txt'

imageListFile1 = '/home/jason6582/sfyc/NUS_Lite/Train_imageOutPutFileList.txt'
tagsFile1 = '/home/jason6582/sfyc/NUS_Lite/Lite_Tags81_Train.txt'

imageListFile2= '/home/jason6582/sfyc/NUS_Lite/Test_imageOutPutFileList.txt'
tagsFile2 = '/home/jason6582/sfyc/NUS_Lite/Lite_Tags81_Test.txt'

jsonFileName1 = 'lite_train.json'
jsonFileName2 = 'lite_test.json'

tags = []
with open(tagListFile, 'r') as f:
    for line in f:
        tags.append(line[:-2])
# tags = sorted(tags, key = str.lower)

annotations = []
availableID = []
count = 0
'''
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
    print count
    print len(annotations)
'''
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
images = []
count = 0
'''
with open(imageListFile1, 'r') as f:
    for line in f:
        if availableID[count] == 1:
            images.append({'id':count, 'file_name': line[:-1]})
        count += 1
    print count
    print len(images)
'''
with open(imageListFile2, 'r') as f:
    for line in f:
        if availableID[count] == 1:
            images.append({'id':count, 'file_name': line[:-1]})
        count += 1
    print count
    print len(images)
for i in range(10):
    print images[i]

caption_data = {'images': images, 'annotations': annotations}
with open(jsonFileName1, 'w') as f:
    json.dump(caption_data, f)
