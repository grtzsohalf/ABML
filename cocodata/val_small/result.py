import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/data/val_small')

result = []

model_type = 'coco'
epoch_num = 1

for e in range(epoch_num):
    epoch = []
    for t in range(4):
        th = float(t) * 0.06 + 0.06
        thres = []
        file_name = 'result_lr0.0003_%s-%s_%s.txt' % (model_type, str(e+50), str(th))
        with open(file_name, 'r') as f:
            for i in range(4):
                f.readline()
            for line in f:
                thres.append(round(float(line.split()[1]), 3))
        epoch.append(thres)
    result.append(epoch)
file_name = 'result_lr0.0003_%s.txt' % model_type
with open(file_name, 'w') as f:
    for e in range(epoch_num):
        f.write('epoch = %s:\n' % str(e+50))
        for t in range(4):
            th = float(t) * 0.06 + 0.06
            f.write('  thres = %s: ' % str(th))
            for i in range(6):
                print e, th, i
                f.write(str(result[e][t][i]) + ' ')
            f.write('\n')
        f.write('\n')
