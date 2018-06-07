import numpy as np
import gensim as gs
from sys import argv
from keras.models import load_model
import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences
import time

seq_length=40
encode_dim=100

t1=time.time()
with open (argv[1], "r") as myfile:
    x_test_temp=myfile.read().splitlines()[1:]

for x in range(len(x_test_temp)):
	ptr=0
	while x_test_temp[x][ptr]!=',': ptr+=1
	x_test_temp[x]=x_test_temp[x][ptr+1:]

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

for x in range(len(x_test_temp)):
	x_test_temp[x]=re.sub('[^a-zA-z0-9\s!?.]','',x_test_temp[x]).lower()
	x_test_temp[x]=elimiate_tripple(x_test_temp[x])

x_test=[]
for line in x_test_temp:
	sentence=[]
	for word in line.split():
		sentence.append(word)
	x_test.append(sentence)
x_test=np.array(x_test)

w2v_model=gs.models.Word2Vec.load('w2v_100_model')
X=np.zeros((x_test.shape[0],seq_length,encode_dim))
for i in range(x_test.shape[0]):
	for j in range(len(x_test[i])):
		if j>(seq_length-1): break
		if x_test[i][j] in w2v_model.wv.vocab:
			X[i,j,:]=w2v_model.wv[x_test[i][j]]

print(time.time()-t1,'seconds used.')

model=load_model('1526651381.1485553_5.h5')
prediction=model.predict(X)

print('predicting...')
o=open(argv[2],'w')
o.write("id,label\n")
for i in range(prediction.shape[0]):
    if prediction[i][0]<0.5:
        o.write("{},{}\n".format(i,0))
    else:
        o.write("{},{}\n".format(i,1))
o.close()
print(time.time()-t1,'seconds used.')
