import gensim as gs
import numpy as np
import time
import re

t1=time.time()
data_labeled=np.load('data_labeled.npy')
data_unlabeled=np.load('data_unlabeled.npy')

corpus=np.concatenate((data_labeled,data_unlabeled))

model = gs.models.Word2Vec(corpus, size=100, window=5, min_count=5, workers=8)
model.save('w2v_100_model')
print('w2v_model saved')
print(model)
print(time.time()-t1,'seconds used.')
