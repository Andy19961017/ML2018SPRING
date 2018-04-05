import sys
import csv as csv
import numpy as np

file1 = open(sys.argv[1] ,"r")
data = csv.reader(file1 , delimiter= ",")
w = np.load('logistic_model.npy')
file2 = open(sys.argv[2],"w+")
output = csv.writer(file2,delimiter=',',lineterminator='\n')
output.writerow(["id","label"])
maxi=np.load('maxi.npy')

para=[]
row_num=0
for row in data :
	if row_num==0:
		row_num+=1
		continue
	for i in range(123):
		para.append(float(row[i]))
	para=np.array(para)/maxi
	para=np.concatenate((np.ones(1),para),axis=0)
	ans=np.dot(para,w)
	if ans>0:
		output.writerow([row_num,1])
	else:
		output.writerow([row_num,0])
	para=[]
	row_num+=1

file1.close()
file2.close()