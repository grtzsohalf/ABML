import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/val')

result = []

model_type = 'mscoco_recursive_concat'
epoch_num = 10

for e in range(epoch_num):
    epoch = []
    e_str = e*1 + 11
    for t in range(1):
        th = float(t) * 0.1 + 0.2
        thres = []
        file_name = 'result_%s-%s_%s.txt' % (model_type, str(e_str), str(th))
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
        e_str = e*1 + 11
        f.write('epoch = %s:\n' % str(e_str))
        for t in range(1):
            th = float(t) * 0.1 + 0.2
            f.write('  thres = %s: ' % str(th))
            for i in range(7):
                f.write(str(result[e][t][i]) + ' ')
            f.write('\n')
        f.write('\n')
