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
1.Accuracy: ~83% (without ensemble).  
2.Ranked 10st out of 116 teams in the Kaggle competition. [Link](https://www.kaggle.com/c/ml-2018spring-hw5/leaderboard).

## Technical Details
### Stage 1: Word Embedding
Details can be referred to [here](https://radimrehurek.com/gensim/models/word2vec.html)

### Stage 2: Sentiment Classification Model  


https://radimrehurek.com/gensim/models/word2vec.html
