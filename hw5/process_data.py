from sys import argv
import numpy as np
import re
import time

def elimiate_tripple(s):
	if len(s)==0: return ""
	to_rem=[]
	for x in range(len(s)-1):
		if s[x]==s[x+1]:
			to_rem.append(x+1)
	re=s[0]
	for x in range(1,len(s)):
		if x not in to_rem:
			re+=s[x]
	return re

t1=time.time()
training_label=[]
with open (argv[1], "r") as myfile:
    data_labeled_temp=myfile.read().splitlines()
with open (argv[2], "r") as myfile:
    data_unlabeled_temp=myfile.read().splitlines()
for x in range(len(data_labeled_temp)):
	training_label.append(data_labeled_temp[x][0])
	data_labeled_temp[x]=re.sub('[^a-zA-z0-9\s!?.]','',data_labeled_temp[x]).lower()
	data_labeled_temp[x]=elimiate_tripple(data_labeled_temp[x])
for x in range(len(data_unlabeled_temp)):
	data_unlabeled_temp[x]=re.sub('[^a-zA-z0-9\s!?.]','',data_unlabeled_temp[x]).lower()
	data_unlabeled_temp[x]=elimiate_tripple(data_unlabeled_temp[x])

data_labeled=[]
for line in data_labeled_temp:
	sentence=[]
	for word in line.split():
		sentence.append(word)
	data_labeled.append(sentence[1:])
data_labeled=np.array(data_labeled)
data_unlabeled=[]
for line in data_unlabeled_temp:
	sentence=[]
	for word in line.split():
		sentence.append(word)
	data_unlabeled.append(sentence)
data_unlabeled=np.array(data_unlabeled)

np.save('data_labeled.npy',data_labeled)
np.save('data_unlabeled.npy',data_unlabeled)
np.save('training_label.npy',np.array(training_label).astype(int))
print(time.time()-t1,'seconds used.')

