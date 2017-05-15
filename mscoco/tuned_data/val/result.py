import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/val')

result = []

model_type = 'mscoco'
epoch_num = 6

for e in range(epoch_num):
    epoch = []
    for t in range(1):
        th = float(t) * 0 + 0.1
        thres = []
        e_str = e*5 + 35
        file_name = 'result_%s-%s.txt' % (model_type, str(e_str))
        with open(file_name, 'r') as f:
            for i in range(4):
                f.readline()
            for line in f:
                thres.append(round(float(line.split()[1]), 3))
        epoch.append(thres)
    result.append(epoch)
file_name = 'result_%s.txt' % model_type
with open(file_name, 'w') as f:
    for e in range(epoch_num):
        e_str = e*5 + 35
        f.write('epoch = %s:\n' % str(e_str))
        for t in range(1):
            th = float(t) * 0 + 0.1
            f.write('  thres = %s: ' % str(th))
            for i in range(7):
                f.write(str(result[e][t][i]) + ' ')
            f.write('\n')
        f.write('\n')
