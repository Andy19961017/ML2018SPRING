Dialog Selection 
==
## Task Description
Given the transcript of a conversation and 5 sentence candidates to choose, the model can choose the sentence that is most likely to appear right next to the conversation.  

For example:
```
How are you today?
Great! And you?

1. The movie is filmed in England
2. Sunday afternoon
3. I'm okay
4. I am a lawyer
5. Actions speak louder than words
```
The answer should be 3. Notice the real problems in this task is in Chinese instead of English.  

## Key Results
1.Accuracy: ~53.5% (without ensemble), ~57.5% (ensemble).  
2.Ranked 1st out of 11 teams in the Kaggle competition. [Link](https://www.kaggle.com/c/ml-2018spring-final-tv-conversation/leaderboard). 

## Key Technical Details
1.Train word embeddings with a large corpus.  
2.Classification model architecture:
  - Input two sentences, the conversation and the sentence right next to it.  
  - LSTM is used encode input sentences.  
  - Cosine similarity is used to compare sentence similarity.  
  - If the two sentences have high cosine similarity, they should appear right next to each other.  

3.Parse training data into positive and negative pairs and train the model. ”Random data processing method” is invented to augment training data, boosting accuracy for 6 %.  
4.Parse the testing data. Pair the diven conversation up with each option (resulting in 5 pairs per problem).  
5.The pair with highest similarity should be the answer.  
6.Detailed description is provided [here](https://github.com/Andy19961017/ML2018SPRING/blob/master/final/Report.pdf). 

## Testing Command
If you want to predict on the given testing_data.csv  
```
bash test.sh dataset/testing_data.csv output_file_path
```
If you want to predict on other testing data  
```
bash test.sh testing_data_path output_file_path
```

## Imported Module
```
keras 2.0.8      
tensorflow 1.4.0      
gensim 3.4.0     
numpy     
jieba     
random     
time     
```
