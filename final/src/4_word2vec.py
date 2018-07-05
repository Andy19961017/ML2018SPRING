import numpy as np
import time
import gensim as gs

t1=time.time()

corpus = np.load('../dataset/corpus_clean_j31.npy')
#corpus2 = np.load('../dataset/corpus_j31.npy')
# test = np.load('../dataset/test_j.npy').reshape(-1)
#corpus = np.load('corpus_j_stop.npy')
#data = np.concatenate((corpus,test))
#data = np.concatenate((corpus2,data))
print(len(corpus),'sentences loaded.')

#print(corpus)

model = gs.models.Word2Vec(corpus,sg=1, size=64, window=7, min_count=2,iter = 200, workers=8)
model.save('../model/wt80_clean_model.model')
print("total time:",time.time()-t1)
