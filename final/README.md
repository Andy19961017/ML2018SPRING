Dialog Selection 
==
## Task Description
Given the transcript of a conversation and 5 sentence candidates to choose, the model can choose the sentence that is most likely to appear right next to the conversation. 

## Key Results
1.Accuracy: ~53.5% (without ensemble), ~57.5% (ensemble).  
2.Ranked 1st out of 11 teams in the Kaggle competition. [Link](https://www.kaggle.com/c/ml-2018spring-final-tv-conversation/leaderboard). 

## Key Technical Details
1.LSTM is used encode input sentences.  
2.Cosine similarity is used to compare sentence similarity. Choose the most similar one.  
2.”Random data processing method” is invented to augment training data, boosting accuracy for 6 %.  
3.Detailed description is provided in [here](https://github.com/Andy19961017/ML2018SPRING/blob/master/final/Report.pdf). 

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
