Text Sentiment Classification 
==
## Task Description
Given a sentence, the model will predict whether it has positive or negative sentiment.  

For example,
```
I won the lottery! ==> positive
I had a bad day... ==> negative
```

## Data
All training and testing data are parsed from Twitter.
  - labeled training data: 200k sentences  
  - unlabeled training data: 1.2M sentences    
  - testing data：200k sentences  

## Key Results
1. Accuracy: ~83.0% (without ensemble).  
2. Ranked 10st out of 116 teams in the Kaggle competition. [Link](https://www.kaggle.com/c/ml-2018spring-hw5/leaderboard).

## Technical Details
### Stage 1: Word Embedding
Details can be referred to [here](https://radimrehurek.com/gensim/models/word2vec.html).  
<p align="center">
<img src="https://github.com/Andy19961017/ML2018SPRING/blob/master/hw5/images/CBOW.png" width="300">
</p>

### Stage 2: Sentiment Classification Model  
#### Choice A: Bag Of Word + DNN, Accuracy: ~76.9%     
<p align="center">
<img src="https://github.com/Andy19961017/ML2018SPRING/blob/master/hw5/images/BOW.png" width="700">
</p>

#### Choice B: GRU + DNN, Accuacy: ~83.0%   
<p align="center">
<img src="https://github.com/Andy19961017/ML2018SPRING/blob/master/hw5/images/GRU.png" width="700">
</p>

### Stage 3: Semi-supervised Training -- Self-training
<p align="center">
<img src="https://github.com/Andy19961017/ML2018SPRING/blob/master/hw5/images/SSL.png" width="700">
</p>

## Discussion: Bag Of Word + DNN  vs  GRU + DNN
In general, GRU is able to consider the order of the words in the sentence while Bag of Word cannot. Therefore, the GRU have higher accuracy.  

Take these two sentences as examples.
```
A: Today is a good day, but it is hot. (negative)
B: Today is hot, but it is a good day. (positive)
```
For Bag of Word model, the two sentences are exactly the same. The sentiment score for them are both 0.66 (0 means negative sentiment and 1 means positive). For GRU model, however, the order of the words is considered. The two sentences are largely different. The sentiment score for them are ```0.15``` for A and ```0.97``` for B.

## Reference
1. Rajat R., Alexis B., Honglak L., Benjamin P., Andrew N. (2007) Self-taught learning: transfer learning from unlabeled data.  
2. https://radimrehurek.com/gensim/models/word2vec.html  
3. https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c  
