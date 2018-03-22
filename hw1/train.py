import csv
import numpy as np
import matplotlib.pyplot as plt

file = open('train.csv', 'r', encoding='big5') 
table = csv.reader(file , delimiter=",")


raw_data=[]
for x in range(18):
	raw_data.append([])

row_num=0
for row in table:
	if row_num==0:
		row_num=row_num+1
		continue
	for x in range(3,27):
		if row[x]=="NR":
			raw_data[(row_num-1)%18].append(float(0))
		else:
			raw_data[(row_num-1)%18].append(float(row[x]))
	row_num=row_num+1

x=[]
for i in range(12*471):
	x.append([])
y=[]


for i in range(12):
	for j in range(471):
		for k in range(18):
			for l in range(9):
				x[i*471+j].append(raw_data[k][i*480+l+j])
			if (k==9): y.append(raw_data[k][i*480+9+j])

x = np.array(x)
y = np.array(y)

bias=np.ones((x.shape[0], 1))
x=np.concatenate((bias,x),axis=1)

w = np.zeros(len(x[0]))
x_trans=x.transpose()

count = 10000
l_rate = 0.0001
iteration=range(count)
RMSE=[]
ada=np.zeros((9*18+1))
for i in range(count):
	y_hat=np.dot(x,w)
	loss=y_hat-y
	cost=np.sum(loss**2)/len(loss)
	cost=np.sqrt(cost)
	grad=2*np.dot(x_trans,loss)
	ada += (grad**2)
	w=w-l_rate*grad/np.sqrt(ada)
	print("count: ",i,' ',"cost: ",cost)
	RMSE.append(cost)
plt.plot(iteration, RMSE, label='learning rate = '+str(l_rate))

l_rate = 0.001
RMSE=[]
w = np.zeros(len(x[0]))
ada=np.zeros((9*18+1))
for i in range(count):
	y_hat=np.dot(x,w)
	loss=y_hat-y
	cost=np.sum(loss**2)/len(loss)
	cost=np.sqrt(cost)
	grad=2*np.dot(x_trans,loss)
	ada += (grad**2)
	w=w-l_rate*grad/np.sqrt(ada)
	print("count: ",i,' ',"cost: ",cost)
	RMSE.append(cost)
plt.plot(iteration, RMSE, label='learning rate = '+str(l_rate))

l_rate = 0.01
RMSE=[]
w = np.zeros(len(x[0]))
ada=np.zeros((9*18+1))
for i in range(count):
	y_hat=np.dot(x,w)
	loss=y_hat-y
	cost=np.sum(loss**2)/len(loss)
	cost=np.sqrt(cost)
	grad=2*np.dot(x_trans,loss)
	ada += (grad**2)
	w=w-l_rate*grad/np.sqrt(ada)
	print("count: ",i,' ',"cost: ",cost)
	RMSE.append(cost)
plt.plot(iteration, RMSE, label='learning rate = '+str(l_rate))

l_rate = 1
RMSE=[]
w = np.zeros(len(x[0]))
ada=np.zeros((9*18+1))
for i in range(count):
	y_hat=np.dot(x,w)
	loss=y_hat-y
	cost=np.sum(loss**2)/len(loss)
	cost=np.sqrt(cost)
	grad=2*np.dot(x_trans,loss)
	ada += (grad**2)
	w=w-l_rate*grad/np.sqrt(ada)
	print("count: ",i,' ',"cost: ",cost)
	RMSE.append(cost)
plt.plot(iteration, RMSE, label='learning rate = '+str(l_rate))


plt.xlabel('iteration')
plt.ylabel('RMSE')
plt.title("Compare Different Learning Rate")
plt.ylim((0,35))
plt.legend()
plt.show()


np.save('7.88279.npy',w)