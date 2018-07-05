import numpy as np
import time
import jieba
from keras.models import load_model
import numpy as np
import gensim as gs
from sys import argv

t1=time.time()

jieba.load_userdict("jieba/dict.txt")

with open ('../'+argv[1], "r") as myfile:
    file=myfile.read().replace("\t", " ").replace("、", "").replace("，", " ").replace(".", "").splitlines()[1:]

parser=[]
for scentence in file:
	ptr=scentence.find(',')
	parser.append(scentence[ptr+1:])

question=[]
for i in range(len(parser)):
	ptr=parser[i].find(',')
	question.append(parser[i][:ptr])
	parser[i]=parser[i][ptr+1:]
# print(question)
# print(len(question))

answer0=[]
answer1=[]
answer2=[]
answer3=[]
answer4=[]
answer5=[]

for i in range(len(parser)):
	answers = parser[i].split(":")
	answer0.append(answers[1][:-1])
	answer1.append(answers[2][:-1])
	answer2.append(answers[3][:-1])
	answer3.append(answers[4][:-1])
	answer4.append(answers[5][:-1])
	answer5.append(answers[6])	
testing_data=np.concatenate((np.array(question).reshape(-1,1),
				np.array(answer0).reshape(-1,1),
				np.array(answer1).reshape(-1,1),
				np.array(answer2).reshape(-1,1),
				np.array(answer3).reshape(-1,1),
				np.array(answer4).reshape(-1,1),
				np.array(answer5).reshape(-1,1)),axis=1)

for i in range(testing_data.shape[0]):
	for j in range(testing_data.shape[1]):
		if testing_data[i,j][-1]!=' ':
			testing_data[i,j]+=' '

# np.save('test.npy',testing_data)
# print(testing_data)

# print(testing_data.shape)

test_jieba=[]
for i in range(len(testing_data)):
	test_jieba.append([])
	for j in range(7):
		sentence = jieba.cut(testing_data[i][j], cut_all=False)
		sentence = list(sentence)
		test_jieba[i].append(sentence)
# print(test_jieba)
# np.save('../dataset/test_clean_j.npy',test_jieba)
# np.save('../dataset/try.npy',test_jieba)

print(time.time()-t1, 'seconds used to parse data.')






def pred(m):
	def scentence_to_mtrx(x):
		encoded=np.zeros((x.shape[0],max_scentence_legnth,word_vec_dim))
		for i in range(x.shape[0]):
			for j in range(len(x[i])):
				if j>(max_scentence_legnth-1): break
				if x[i][-(j+1)] in word_vecs.wv.vocab:
					encoded[i,-(j+1),:]=word_vecs.wv[x[i][-(j+1)]]
		return encoded
	
	global model_count
	global max_scentence_legnth
	global test_encoded

	model=load_model('../model/'+m)
	if model_count==0:
		max_scentence_legnth=model.input[0].shape[1]
		test_encoded=np.zeros((test.shape[0], test.shape[1], max_scentence_legnth, word_vec_dim))
		for i in range(7):
			test_encoded[:,i]=scentence_to_mtrx(test[:,i])
	else:
		if max_scentence_legnth!=model.input[0].shape[1]:
			max_scentence_legnth=model.input[0].shape[1]
			test_encoded=np.zeros((test.shape[0], test.shape[1], max_scentence_legnth, word_vec_dim))
			for i in range(7):
				test_encoded[:,i]=scentence_to_mtrx(test[:,i])
	model_count+=1
	print('\n=========model '+m+'===========')
	print('seqence length:',max_scentence_legnth)

	predictions=[]
	for i in range(1,7):
		pred=model.predict([test_encoded[:,0], test_encoded[:,i]])
		predictions.append(pred)
	prediction=np.argmax(np.array(predictions), axis=0)
	# print(prediction.shape)
	return prediction

t1=time.time()
model_count=0
max_scentence_legnth=0
test_encoded=0
test=np.array(test_jieba)


word_vecs=gs.models.Word2Vec.load('../model/wt64_clean_model.model')
word_vec_dim=word_vecs.wv.syn0.shape[1]

ensemble_pred=[]
for m in argv[3:]:
	t2=time.time()
	ensemble_pred.append(pred(m))
	print(time.time()-t2, 'seconds used.')

# print(ensemble_pred.shape)
ensemble_pred=np.concatenate(tuple(ensemble_pred),axis=1)
# print(ensemble_pred.shape)
# print(prediction.shape)
final_pred=[]
for i in range(ensemble_pred.shape[0]):
	counts = np.bincount(ensemble_pred[i])
	final_pred.append(np.argmax(counts))

file='../'+argv[2]
o=open(file,'w')
o.write("id,ans\n")
for i in range(len(final_pred)):
	o.write("{},{}\n".format(i,final_pred[i]))
o.close()
print(argv[2],'saved.')
print(model_count,'models used.')
print(time.time()-t1, 'seconds used in total.')
