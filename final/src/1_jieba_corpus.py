# -*- coding: utf-8 -*-
import jieba
import numpy as np

jieba.load_userdict("dict.txt")
word_list = []

windownum = 1
stride = 1
word_list2 = []
with open ("../dataset/training_data/1_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
for i in range(len(content)):
	sentence = jieba.cut(content[i].replace('.', '').replace('\"', '').replace('，', ' ').replace('「', '').replace('」', '').replace('、', ''), cut_all=False)
	sentence = list(sentence)
	#print(sentence)
	word_list.append(sentence)
for i in range(0,(len(word_list)-windownum+1),stride):
    sentence_comb=[]
    for j in range(windownum):
        sentence_comb+=word_list[i+j]
        sentence_comb+=[' ']
    word_list2.append(sentence_comb)
word_list=[]


with open ("../dataset/training_data/2_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
for i in range(len(content)):
	sentence = jieba.cut(content[i].replace('.', '').replace('\"', '').replace('，', ' ').replace('「', '').replace('」', '').replace('、', ''), cut_all=False)
	sentence = list(sentence)
	word_list.append(sentence)
for i in range(0,(len(word_list)-windownum+1),stride):
    sentence_comb=[]
    for j in range(windownum):
        sentence_comb+=word_list[i+j]
        sentence_comb+=[' ']
    word_list2.append(sentence_comb)
word_list=[]


with open ("../dataset/training_data/3_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
for i in range(len(content)):
	sentence = jieba.cut(content[i].replace('.', '').replace('\"', '').replace('，', ' ').replace('「', '').replace('」', '').replace('、', ''), cut_all=False)
	sentence = list(sentence)
	word_list.append(sentence)
for i in range(0,(len(word_list)-windownum+1),stride):
    sentence_comb=[]
    for j in range(windownum):
        sentence_comb+=word_list[i+j]
        sentence_comb+=[' ']
    word_list2.append(sentence_comb)
word_list=[]


with open ("../dataset/training_data/4_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
for i in range(len(content)):
	sentence = jieba.cut(content[i].replace('.', '').replace('\"', '').replace('，', ' ').replace('「', '').replace('」', '').replace('、', ''), cut_all=False)
	sentence = list(sentence)
	word_list.append(sentence)
for i in range(0,(len(word_list)-windownum+1),stride):
    sentence_comb=[]
    for j in range(windownum):
        sentence_comb+=word_list[i+j]
        sentence_comb+=[' ']
    word_list2.append(sentence_comb)
word_list=[]

with open ("../dataset/training_data/5_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
for i in range(len(content)):
	sentence = jieba.cut(content[i].replace('.', '').replace('\"', '').replace('，', ' ').replace('「', '').replace('」', '').replace('、', ''), cut_all=False)
	sentence = list(sentence)
	word_list.append(sentence)
for i in range(0,(len(word_list)-windownum+1),stride):
    sentence_comb=[]
    for j in range(windownum):
        sentence_comb+=word_list[i+j]
        sentence_comb+=[' ']
    word_list2.append(sentence_comb)
word_list=[]

print(word_list2)
print("sentence_num:",len(word_list2))
np.save('../dataset/corpus_clean_j11.npy',word_list2)



'''
with open ("dataset/training_data/1_train.txt", "r") as myfile:
    content=myfile.read().splitlines()
words = jieba.cut(content, cut_all=False)
word_list = list(words)
for i in range(100000):
	if ('\n') not in word_list:
   		break;
	word_list.remove('\n')

print(word_list)
np.save('corpus.npy',word_list)


'''

'''
# 我們要從answers中挑出應該接在dialogue之後的短句
dialogue = "如果飛機在飛行當中打一個小洞的話 會不會影響飛行的安全呢"
answers = [
  "其實狗搖尾巴有很多種方式 高興搖尾巴 生氣也搖尾巴",  
  "如果這個洞的話經過仔細的設計的話 應該不至於造成太大問題",
  "所以只要依照政府規定 在採收前十天不要噴灑農藥", 
  "靜電才是加油站爆炸的元凶 手機不過是代罪羔羊",
  "我們可以用表面張力及附著力的原理 來測試看看",
  "不過蝦子死亡後 身體會釋放出有毒素的體液 可能造成水的變質"]

emb_cnt = 0
avg_dlg_emb = np.zeros((dim,))
# jieba.cut 會把dialogue作分詞
# 對於有在word_vecs裡面的詞我們才把它取出
# 最後詞向量加總取平均，作為句子的向量表示
for word in jieba.cut(dialogue):
  if word in word_vecs:
    avg_dlg_emb += word_vecs[word]
    emb_cnt += 1
avg_dlg_emb /= emb_cnt

emb_cnt = 0
max_idx = -1
max_sim = -10
# 在六個回答中，每個答句都取詞向量平均作為向量表示
# 我們選出與dialogue句子向量表示cosine similarity最高的短句
for idx,ans in enumerate(answers):
  avg_ans_emb = np.zeros((dim,))
  for word in jieba.cut(ans):
    if word in word_vecs:
      avg_ans_emb += word_vecs[word]
      emb_cnt += 1
  sim = np.dot(avg_dlg_emb, avg_ans_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_ans_emb)
  print("Ans#%d: %f" % (idx, sim))
  if sim > max_sim:
    max_idx = idx
    max_sim = sim

print("Answer:%d" % max_idx)
'''