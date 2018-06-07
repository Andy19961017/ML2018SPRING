import numpy as np # linear algebra
import gensim as gs
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import ModelCheckpoint
import time

num='5'
print('2 layer bi-direct')
seq_length=40
encode_dim=100
GRU_out_dim=128
validation = 0.1
epoch=1000000
batch_size = 512

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

t1=time.time()
data_labeled=np.load('data_labeled.npy')
y_train=np.load('training_label.npy')
w2v_model=gs.models.Word2Vec.load('w2v_100_model')

x_train=np.zeros((data_labeled.shape[0],seq_length,encode_dim))
for i in range(data_labeled.shape[0]):
	for j in range(len(data_labeled[i])):
		if j>(seq_length-1): break
		if data_labeled[i][j] in w2v_model.wv.vocab:
			x_train[i,j,:]=w2v_model.wv[data_labeled[i][j]]

model = Sequential()
model.add(GRU(GRU_out_dim, input_shape=(seq_length,encode_dim), return_sequences=True, dropout=0.2, recurrent_dropout=0.2, go_backwards=True))
model.add(GRU(GRU_out_dim, dropout=0.2, recurrent_dropout=0.2, go_backwards=True))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

t=str(time.time())
checkpoint= ModelCheckpoint(t+'_'+num+'.h5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')
model.fit(x_train, y_train, epochs = epoch, batch_size=batch_size, validation_split=validation, verbose = 1, callbacks=[checkpoint])
model.save('1526651381.1485553_5.h5')
print('1526651381.1485553_5.h5 saved.')
print(time.time()-t1,'seconds used.')
