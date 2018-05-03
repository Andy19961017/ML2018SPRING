import sys
from sklearn.cluster import *
from keras.models import load_model
import numpy as np
import time

encoder=load_model('TAencoder.h5')
X=np.load(sys.argv[1])
X=X.astype('float32')/255.
X=np.reshape(X,(len(X),-1))
encoded_imgs=encoder.predict(X)
encoded_imgs=encoded_imgs.reshape(encoded_imgs.shape[0],-1)
kmeans=KMeans(n_clusters=2,random_state=0).fit(encoded_imgs)


import pandas as pd
f=pd.read_csv(sys.argv[2])
IDs,idx1,idx2=np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

o=open(sys.argv[3],'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs,idx1,idx2):
	a=kmeans.labels_[i1]
	b=kmeans.labels_[i2]
	if a==b:
		pred=1
	else:
		pred=0
	o.write("{},{}\n".format(idx,pred))
o.close()
