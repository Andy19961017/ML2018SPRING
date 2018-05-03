import sys
import csv
import numpy
import pandas
import keras
from keras.models import load_model

model1 = sys.argv[1]
model2 = sys.argv[2]
model3 = sys.argv[3]

test_file='../data/test.csv'
predict_file="../prediction/ensemble_predict.csv"

temp=pandas.read_csv(test_file,sep=',').values
x_test=numpy.zeros((temp.shape[0],48*48),dtype=float)
for i in range(x_test.shape[0]):
    x_test[i]=numpy.asarray(temp[i,1].split(' '),dtype=float)
x_test=x_test.reshape(-1,48,48,1)/255.0

print( 'Load Model:',model1,model2,model3 )
model1 = load_model( '../model/'+model1 )
model2 = load_model( '../model/'+model2 )
model3 = load_model( '../model/'+model3 )
label1 = model1.predict( x_test )
label2 = model2.predict( x_test )
label3 = model3.predict( x_test )
print('l1.shape',label1.shape)
label=numpy.concatenate((label1,label2,label3),axis=1)
print('l.shape',label.shape)
label=numpy.argmax(label,axis=1)
print('l.shape',label.shape)
label=label%7
print('l.shape',label.shape)

# output
file = open(predict_file,"w+")
output = csv.writer(file,delimiter=',',lineterminator='\n')
output.writerow(["id","label"])
for i in range(label.shape[0]):
        output.writerow([i, label[i]])
file.close()
