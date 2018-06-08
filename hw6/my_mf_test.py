from keras.models import load_model
import keras.backend as K
import numpy as np
import csv
import os
from sys import argv

def read_test(file_name):
    data=[]
    with open(file_name, 'r') as f:
        f.readline()
        reader=csv.reader(f)
        for row in reader:
            data_id, usr_id, mov_id=row
            data.append( [data_id, int(usr_id), int(mov_id)] )
    return np.array(data)
def preprocess(data, genders, ages, occ, movs):
    if data.shape[1] == 4:
        np.random.seed(1019)
        index=np.random.permutation(len(data))
        data=data[index]
    usr_id=np.array(data[:,1],dtype=int)
    mov_id=np.array(data[:,2],dtype=int)
    usrGdr=np.array(genders)[usr_id]
    usrAge=np.array(ages)[usr_id]
    usrOccu=np.array(occ)[usr_id]
    movGenre=np.array(movs)[mov_id]
    std=np.std(usrAge)
    usrAge=usrAge/std
    score=[]
    if data.shape[1]==4:
        score=data[:,3].reshape(-1,1)
    return usr_id,mov_id,usrGdr,usrAge,usrOccu,movGenre,score

def to_catel(index, cate):
    catel = np.zeros(cate, dtype=int)
    catel[index] = 1
    return list(catel)

def read_mov(file_name):
    def genre_to_num(genr, all_genr):
        result=[]
        for g in genr.split('|'):
            if g not in all_genr:
                all_genr.append(g)
            result.append( all_genr.index(g) )
        return result, all_genr
    movs, all_genr=[[]] * 3953, []
    with open(file_name, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            mov_id, title, genre=line[:-1].split('::')
            genre_num, all_genr=genre_to_num(genre, all_genr)
            movs[int(mov_id)]=genre_num
    cate=len(all_genr)
    for i, m in enumerate(movs):
        movs[i]=to_catel(m, cate)
    return movs, all_genr

def read_usr(file_name):
    genders, ages, occ=[[]]*6041, [[]]*6041, [ [0]*21 ]*6041
    cate=21
    with open(file_name, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            usr_id, gender, age, occu, zipcode=line[:-1].split('::')
            genders[int(usr_id)]=0 if gender is 'F' else 1
            ages[int(usr_id)]=int(age)
            occ[int(usr_id)]=to_catel(int(occu), cate)
    return genders, ages, occ

gndr,age,occu=read_usr(argv[4])
movies,gen=read_mov(argv[3])
test=read_test(argv[1])
_id=np.array(test[:,0]).reshape(-1,1)
userid,movieid,userGdr,userAge,userOccu,movieGenre,_Y=preprocess(test,gndr,age,occu,movies)
def rmse(y_true, y_pred):
	return K.sqrt(K.mean((y_pred-y_true)**2))

model=load_model('mf_model.h5',custom_objects={'rmse':rmse})
result=model.predict([userid,movieid,userGdr,userAge,userOccu,movieGenre])
score=np.clip(result,1,5).reshape(-1,1)
output=np.array(np.concatenate((_id,score),axis=1))
with open(argv[2],'w') as f:
	writer = csv.writer(f)
	writer.writerow(['TestDataID', 'Rating'])
	writer.writerows(output)