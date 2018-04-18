import sys
import csv
import numpy
import pandas
import keras
from keras.models import load_model

height = 48
width  = 48
channels = 1
category_count = 7

temp = pandas.read_csv( sys.argv[1], sep=',').values
x_test = numpy.zeros( ( temp.shape[0], height * width), dtype=float)
for i in range(x_test.shape[0]):
    x_test[i] = numpy.asarray( temp[i, 1].split(' '), dtype=float )
x_test = x_test.reshape(-1, height, width, channels) / 255.0

model = load_model( sys.argv[3] )
label = model.predict( x_test )

file = open(sys.argv[2],"w+")
output = csv.writer(file,delimiter=',',lineterminator='\n')
output.writerow(["id","label"])
for i in range(label.shape[0]):
        output.writerow([i, pandas.Series(label[i]).idxmax()])
file.close()
