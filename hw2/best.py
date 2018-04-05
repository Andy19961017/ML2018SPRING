import sys
import csv as csv
import numpy as np

append_name="logistic_tri"
continuous=[0,10,78,79,80]
tri=[0,78,79,80]

file1 = open(sys.argv[1] ,"r")
data = csv.reader(file1 , delimiter= ",")
# w = np.load(append_name+'_3000'+'_model.npy')
w = np.load('best_model.npy')
file2 = open(sys.argv[2],"w+")
output = csv.writer(file2,delimiter=',',lineterminator='\n')
output.writerow(["id","label"])
maxi=np.load('3maxi.npy')
# mini=np.load('mini.npy')

para=[]
row_num=0
for row in data :
	if row_num==0:
		row_num+=1
		continue
	for i in range(len(row)):
		para.append(float(row[i]))
		if i in continuous:
			para.append(float(row[i])**2)
			if i in tri:
				para.append(float(row[i])**3)
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