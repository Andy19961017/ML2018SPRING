# for data parsing
import sys
import pandas
from keras.utils import np_utils  
import numpy as np  
np.random.seed(10)
# for learning
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import time

# for plotting
#import matplotlib.pyplot as plt  
# import os

height = 48
width  = 48
channel = 1
category_num = 7
shape=(height,width,channel)

valid=0
data_aug=True

batch_size=100
epochs=1000

use_work_station = True

# GPU option
if use_work_station:
    import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# record used time
t1 = time.time()

# parse data
temp = pandas.read_csv( sys.argv[1], sep=',').values
y_train = np.copy( (temp[:,0]).astype(np.int16) )
y_train = np_utils.to_categorical( y_train, num_classes=category_num)
x_train = np.zeros( ( temp.shape[0], height * width), dtype=float)
for i in range(x_train.shape[0]):
    x_train[i] = np.asarray( temp[i, 1].split(' '), dtype=float )
x_train = x_train.reshape( -1, height, width, channel)
x_train/=255
del temp
#mean, std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
#np.save('attr.npy', [mean, std])
#x_train = (x_train - mean) / (std + 1e-20)

print('=========Parsed Data=========')
print('x_train(batch, height, width, channel):', x_train.shape)
print('y_train(batch, num of classes):', y_train.shape)

if valid != 0:
  valid_num = int(x_train.shape[0]*valid)
  x_valid, y_valid = x_train[-valid_num:], y_train[-valid_num:]
  x_train, y_train = x_train[:-valid_num], y_train[:-valid_num]

# Build Model
model=Sequential()

model.add(Conv2D(70, kernel_size=(5, 5), input_shape=shape, padding='same'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(130, kernel_size=(3, 3), padding='same'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(500, kernel_size=(3, 3), padding='same'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(500, kernel_size=(3, 3), padding='same'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary()) 

# train
t_str=time.ctime().replace(' ','_')

train_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=[0.8, 1.2],
                              horizontal_flip=True)
train_gen.fit(x_train)
# valid_gen = ImageDataGenerator(rotation_range=30,
#                                 width_shift_range=0.2,
#                                 height_shift_range=0.2,
#                                 zoom_range=[0.8, 1.2],
#                                 shear_range=0.2,
#                                 horizontal_flip=True)
# valid_gen.fit(x_valid)

checkpoint= ModelCheckpoint('../model/'+t_str+'.h5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')

if valid!=0:
  model.fit_generator(train_gen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=(x_valid,y_valid),
                    callbacks=[checkpoint])
else:
  model.fit_generator(train_gen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    callbacks=[checkpoint])

model.save('model.h5')
# show runtime
t2 = time.time()
print("Process time: %.1f hr, %.1f min, %.1f sec." % \
((int((t2-t1)//3600)), (int((t2-t1)//60)), ((t2-t1)%60)))

sys.stdout.flush()
