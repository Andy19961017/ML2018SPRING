import sys
from os import listdir
from skimage import io, img_as_uint
import numpy as np
import time
t1=time.time()
a=sys.argv[1]
b=sys.argv[2]

data=[]
print("Reading data...")
for pic in listdir(a):
	data.append(io.imread(a+'/'+pic).reshape(-1))
print(len(data),"picture loaded.")
data=np.array(data)
mean=np.mean(data,axis=0)
# io.imsave('mean_face.jpg',mean.reshape(600,600,3).astype(np.uint8))
data=np.subtract(data,mean)
data=data.T
print("SVD...")
U,S,V=np.linalg.svd(data,full_matrices=False)
target_face=io.imread(a+'/'+b).reshape(1,-1).astype(np.float64)
print("Reconstructing...")

target_face-=mean
k=4
weight=np.dot(target_face,U[:,:k])
target_face=np.zeros((1,600*600*3))
for x in range(k):
    target_face+=weight[0,x]*U[:,x].T
target_face+=mean
target_face-=np.min(target_face)
target_face/=np.max(target_face)
target_face=(target_face*255).astype(np.uint8)
io.imsave('reconstruction.jpg',target_face.reshape(600,600,3))
print("reconstruction.jpg saved.")
print(time.time()-t1,"sec")
