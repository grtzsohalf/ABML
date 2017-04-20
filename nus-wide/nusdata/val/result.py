import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/data/val_small')

result = []

model_type = 'nus_noatt'
epoch_num = 1

for e in range(epoch_num):
    epoch = []
    for t in range(1):
        th = float(t) * 0 + 0.3
        thres = []
        file_name = 'result_%s-%s_%s.txt' % (model_type, str(e+4), str(th))
        with open(file_name, 'r') as f:
            for i in range(4):
                f.readline()
            for line in f:
                thres.append(round(float(line.split()[1]), 4))
        epoch.append(thres)
    result.append(epoch)
file_name = 'result_%s.txt' % model_type
with open(file_name, 'w') as f:
    for e in range(epoch_num):
        f.write('epoch = %s:\n' % str(e+4))
        for t in range(1):
            th = float(t) * 0 + 0.3
            f.write('  thres = %s: ' % str(th))
            for i in range(7):
                f.write(str(result[e][t][i]) + ' ')
            f.write('\n')
        f.write('\n')
