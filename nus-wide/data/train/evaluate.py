import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow')

from core.utils import *
# print reference
refsNum = 0
cansNum = 0
correctNum = 0
LEN = 3
# f = open('top30_ans.txt', 'w')
# for i in range(len(candidate)):
# for i in range(30):
for part in range(16):
    candidate = load_pickle('train.candidate.captions81_mask_%s.pkl' %(str(part)))
    reference = load_pickle('train.references81_%s.pkl' %(str(part)))
    for i in range(len(candidate)):
        refs = str(reference[i][0][:-2]).split()
        refsNum += len(refs)
        cans = str(candidate[i][:]).split()
        cansNum += LEN
        refsDict = {}
        correct = 0
        print "Instance", (i+1), ":"
        print "Predicted: ", cans[:LEN]
        print "Ground truth: ", refs
        for j in range(LEN):
            c = cans[j]
            refsDict[c] = 0
        for r in refs:
            refsDict[r] = 1
        for j in range(LEN):
            c = cans[j]
            if refsDict[c] == 1:
                correct += 1
        correctNum += correct
recall = float(correctNum)/float(refsNum)
precision = float(correctNum)/float(cansNum)

g = open('result.txt', 'w')
g.write('Total number of true label: ' + str(refsNum) + '\n')
g.write('Total number of predict label: ' + str(cansNum) + '\n')
g.write('Number of correct prediction: ' + str(correctNum) + '\n\n')
g.write('Recall: ' + str(recall) + '\n')
g.write('Precision: ' + str(precision) + '\n')

# val_data = load_coco_data(data_path='../../data', split= 'val')
# reference = load_pickle('val.references81.pkl')
# candidate = load_pickle('val.candidate.captions.pkl')
# fileName = val_data['file_names']
#for i in range(len(fileName)):
 #   print fileName[i]
#for i in range(100):
 #   print reference[i]
  #  print candidate[i]
