import csv
import numpy as np
import sys

file1 = open(str(sys.argv[1]),"r")
# file1 = open("test.csv","r")
data = csv.reader(file1 , delimiter= ",")
w = np.load('6.16044.npy')
file2 = open(str(sys.argv[2]),"w+")
# file2 = open('best_predict.csv',"w+")
output = csv.writer(file2,delimiter=',',lineterminator='\n')
output.writerow(["id","value"])

dic={ "AMB_TEMP":0, "CH4":1, "CO":2, "NMHC":3, "NO":4, "NO2":5, "NOx":6, 
	"O3":7, "PM10":8, "PM2.5":9, "RAINFALL":10, "RH":11, "SO2":12,
	 "THC":13, "WD_HR":14, "WIND_DIREC":15, "WIND_SPEED":16, "WS_HR":17}
# used_feat=range(18);
# used_feat=[ dic["CO"], dic["NMHC"], dic["NO2"], dic["NOx"], dic["PM10"],
	 # dic["PM2.5"] , dic["SO2"], dic["THC"]]
used_feat=[ dic["CO"], dic["NO2"], dic["PM10"], dic["PM2.5"]]
# used_feat=[dic["PM10"], dic["PM2.5"]]
# used_feat=[dic["PM2.5"]]
diff_thresh=30
used_hour=9

def process_data1(para):
	for x in range(len(used_feat)):
		for i in range(1,used_hour):
			if para[x*9+i]<=0:
				para[x*9+i]=para[x*9+i-1]
		for i in range(used_hour-2,-1):
			if para[x*9+i]<=0:
				para[x*9+i]=para[x*9+i+1]

def process_data2(data):
	for x in range(len(used_feat)):
		for i in range(1,used_hour-1):
			a=abs(data[i+used_hour*x]-data[i+used_hour*x-1])
			a+=abs(data[i+used_hour*x]-data[i+used_hour*x+1])
			if a>diff_thresh:
				data[i+used_hour*x]=(data[i+used_hour*x-1]+data[i+used_hour*x+1])/2
		for i in range(used_hour-2,0):
			a=abs(data[i+used_hour*x]-data[i+used_hour*x-1])
			a+=abs(data[i+used_hour*x]-data[i+used_hour*x+1])
			if a>diff_thresh:
				data[i+used_hour*x]=(data[i+used_hour*x-1]+data[i+used_hour*x+1])/2

def process_data(para):
	process_data1(para)
	# process_data2(para)
	return None

para=[]
row_num=0
for row in data :
	if not (row_num%18 in used_feat):
		row_num=row_num+1
		continue
	for i in range(2,11):
		if row[i]=="NR":
			para.append(float(0))
		else: 
			para.append(float(row[i]))
	if row_num%18==max(used_feat):
		process_data(para)
		para=np.array(para)
		para=np.concatenate((np.ones(1),para),axis=0)
		ans=np.dot(para,w)
		output.writerow(["id_"+str((row_num-1)//18),ans])
		para=[]
	row_num+=1

file1.close()
file2.close()