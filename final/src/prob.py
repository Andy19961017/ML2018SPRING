from random import randint
import numpy as np

import collections
import matplotlib.pyplot as plt


jump=1
multiple_wrong=1

#q_len=np.arange(5,23)
#a_len=np.arange(3,11)

def get_prob(p):
	if(p>1 or p<0):
		print("p range error")
		return False
	rand = randint(1,100)
	p = p*100
	if(rand>p):
		return False
	else:
		return True


corpus=np.load('dataset/corpus_clean_j11.npy')
print('corpus:')
print(corpus)
print(corpus.shape)

corp_len=corpus.shape[0]

right_ques=[]
right_ans=[]
wrong_ques=[]
wrong_ans=[]

for i in range(corp_len-10):
	q = corpus[i]
	tmp = 1
	while True:
		if len(q) <6 :
			if(get_prob(0.65)):
				q += corpus[i+tmp]
				tmp+=1
			else:
				break
		elif len(q) <10:
			if(get_prob(0.55)):
				q += corpus[i+tmp]
				tmp+=1
			else:
				break		
		elif len(q) <13:
			if(get_prob(0.35)):
				q += corpus[i+tmp]
				tmp+=1
			else:
				break
		elif len(q) <17:
			if(get_prob(0.25)):
				q += corpus[i+tmp]
				tmp+=1
			else:
				break		
		elif len(q) <20:
			if(get_prob(0.05)):
				q += corpus[i+tmp]
				tmp+=1
			else:
				break
		else:
			break
	

	a = corpus[i+tmp]
	if len(a)<6:
		if(get_prob(0.3)):
			a+= corpus[i+tmp+1]
	right_ques.append(q.copy())
	right_ans.append(a.copy())
	if(i<5):
		print("q:",q)	
		print("a:",a)
		print(right_ans)



	for t in range(multiple_wrong):
		wrong_ques.append(q.copy())
		wrong_idx=i+randint(100, 300000)
		a = corpus[wrong_idx%corp_len]
		if len(a)<6:
			if(get_prob(0.3)):
				a+= corpus[(wrong_idx+1)%corp_len]
		wrong_ans.append(a.copy())

right_ques=np.array(right_ques)
right_ans=np.array(right_ans)
wrong_ques=np.array(wrong_ques)
wrong_ans=np.array(wrong_ans)

print(right_ques.shape)
print(right_ans.shape)
print(wrong_ques.shape)
print(wrong_ans.shape)

for i in range(4):
	print(right_ques[i])
	print(right_ans[i])
	print(wrong_ques[i])
	print(wrong_ans[i])



l= [len(x) for x in right_ques ]
m = max( l )
a = np.average( l )
print('max len of right_ques:',m)
print('average len of right_ques:',a)
d=collections.Counter(l)
print(collections.OrderedDict(sorted(d.items())),'\n')
plt.hist(l)
plt.show()

l= [len(x) for x in right_ans ]
m = max( l )
a = np.average( l )
print('max len of right_ans:',m)
print('average len of right_ans:',a)
d=collections.Counter(l)
print(collections.OrderedDict(sorted(d.items())),'\n')
plt.hist(l)
plt.show()

l= [len(x) for x in wrong_ques ]
m = max( l )
a = np.average( l )
print('max len of wrong_ques:',m)
print('average len of wrong_ques:',a)
d=collections.Counter(l)
print(collections.OrderedDict(sorted(d.items())),'\n')
plt.hist(l)
plt.show()


l= [len(x) for x in wrong_ans ]
m = max( l )
a = np.average( l )
print('max len of wrong_ans:',m)
print('average len of wrong_ans:',a)
d=collections.Counter(l)
print(collections.OrderedDict(sorted(d.items())),'\n')
plt.hist(l)
plt.show()


right_pair=np.concatenate((right_ques.reshape(-1,1),right_ans.reshape(-1,1)),axis=1)
wrong_pair=np.concatenate((wrong_ques.reshape(-1,1),wrong_ans.reshape(-1,1)),axis=1)
print('right:')
print(right_pair.shape)
print(right_pair[0])
print(right_pair[1])
print(right_pair[2])

print('wrong:')
print(wrong_pair.shape)
print(wrong_pair[0])
print(wrong_pair[1])
print(wrong_pair[2])

label=np.concatenate((np.ones(right_pair.shape[0]),np.zeros(wrong_pair.shape[0])))

pair=np.concatenate((right_pair,wrong_pair),axis=0)
final=np.concatenate((pair,label.reshape(-1,1)),axis=1)
print('right:')
print(final.shape)
print(final[0])
print(final[1])
print(final[2])

print('wrong:')
print(final.shape)
print(final[right_pair.shape[0]])
print(final[right_pair.shape[0]+1])
print(final[right_pair.shape[0]+2])

np.save('dataset/training_pair_corpus_j_prob_clean'+'.npy', final)
print('dataset/training_pair_corpus_j_prob_clean'+'.npy saved')




'''
##改上面的時候for裡面記得手動調
for i in range(0, corpus.shape[0]-q_len-a_len+1, jump):
	q=corpus[i]+corpus[i+1]+corpus[i+2]+corpus[i+3]
	right_ques.append(q)
	right_ans.append(corpus[i+4]+corpus[i+5])
	for t in range(multiple_wrong):
		wrong_ques.append(q)
		wrong_idx=i+randint(100, 300000)
		wrong_ans.append(corpus[wrong_idx%corp_len]+corpus[(wrong_idx+1)%corp_len])
'''