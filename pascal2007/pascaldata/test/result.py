import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/data/val_small')

result = []

model_type = 'pascal_0.002'
epoch_num = 5

for e in range(epoch_num):
    epoch = []
    e = 300 + e*50
    for t in range(1):
        th = float(t) * 0 + 0.09
        thres = []
        file_name = 'result_%s-%s_%s.txt' % (model_type, str(e), str(th))
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
        f.write('epoch = %s:\n' % str(e))
        for t in range(1):
            th = float(t) * 0 + 0.09
            f.write('  thres = %s: ' % str(th))
            for i in range(7):
                f.write(str(result[e][t][i]) + ' ')
            f.write('\n')
        f.write('\n')
