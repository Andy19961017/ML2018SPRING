import csv
import numpy as np
import sys

file1 = open(str(sys.argv[1]) ,"r")
data = csv.reader(file1 , delimiter= ",")
w = np.load('7.88279.npy')
file2 = open(str(sys.argv[2]),"w+")
output = csv.writer(file2,delimiter=',',lineterminator='\n')
output.writerow(["id","value"])

para=[]
row_num=0
for row in data :
	for i in range(2,11):
		if row[i]=="NR":
			para.append(float(0))
		else: 
			para.append(float(row[i]))
	if row_num%18==17:
		para=np.array(para)
		para=np.concatenate((np.ones(1),para),axis=0)
		ans=np.dot(para,w)
		output.writerow(["id_"+str((row_num-1)//18),ans])
		para=[]
	row_num+=1

file1.close()
file2.close()


