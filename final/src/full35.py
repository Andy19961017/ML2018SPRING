import keras
from keras.layers import Input, LSTM, Dense, Dot, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import gensim as gs
import numpy as np
from sys import argv
import time

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

word_vec_dim=64
rnn_dim=512
batch_size=512
epochs=100000
validation=0.1

data=np.load('../data/training_pair_corpus_j_prob_clean.npy')
w2v_model=gs.models.Word2Vec.load('../model/wt64_clean_model.model')

max_scentence_legnth = max( [len(x) for x in data[:,0] ] + [len(x) for x in data[:,1] ])
print('max_scentence_legnth:',max_scentence_legnth)

if validation!=0:
	data_train, data_valid=train_test_split(data,test_size=validation,shuffle=True)

data_train=data_train
data_valid=data_valid

def scentence_to_mtrx(x):
	encoded=np.zeros((x.shape[0],max_scentence_legnth,word_vec_dim))
	for i in range(x.shape[0]):
		for j in range(len(x[i])):
			if j>(max_scentence_legnth-1): break
			if x[i][-(j+1)] in w2v_model.wv.vocab:
				encoded[i,-(j+1),:]=w2v_model.wv[x[i][-(j+1)]]
	return encoded

def generate_batch(data, batch_size):
    loopcount = data.shape[0] // batch_size
    i=0
    while (True):
        yield [scentence_to_mtrx(data[i*batch_size:(i+1)*batch_size,0]), scentence_to_mtrx(data[i*batch_size:(i+1)*batch_size,1])] , data[i*batch_size:(i+1)*batch_size,2]
        i+=1
        if i==loopcount:i=0

def get_model():
	question = Input(shape=(max_scentence_legnth, word_vec_dim))
	answer = Input(shape=(max_scentence_legnth, word_vec_dim))

	shared_lstm = LSTM(rnn_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, go_backwards=True)
	encoded_ques = shared_lstm(question)
	encoded_ans = shared_lstm(answer)
	shared_lstm = LSTM(rnn_dim, dropout=0.2, recurrent_dropout=0.2, go_backwards=True)
	encoded_ques = shared_lstm(encoded_ques)
	encoded_ans = shared_lstm(encoded_ans)
	encoded_ques = Dropout(0.2)(encoded_ques)
	encoded_ans = Dropout(0.2)(encoded_ans)
	sim=Dot(axes=1, normalize=True)([encoded_ques, encoded_ans])
	sim=BatchNormalization()(sim)
	sim=Activation('sigmoid')(sim)
	model = Model(inputs=[question, answer], outputs=sim)
	model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
	print(model.summary())
	return model


t=str(time.time())
checkpoint= ModelCheckpoint('../model/'+argv[0]+'_'+t+'.h5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')
# print(data_train.shape)
# print(data_train)
# print(data_valid.shape)
# print(data_valid)
model=get_model()
model.fit_generator(
	generator=generate_batch(data_train, batch_size),                                                      
    steps_per_epoch=data_train.shape[0]//batch_size,
    nb_epoch=epochs, 
    validation_data=generate_batch(data_valid,batch_size),
    validation_steps=data_valid.shape[0]//batch_size, 
    verbose=1,
    callbacks=[checkpoint])
model.save('../model/'+argv[0]+'_'+t+'_final'+'.h5')
print('../model/'+argv[0]+'_'+t+'_final'+'.h5 saved.')

