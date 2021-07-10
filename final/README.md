Dialog Selection 
==
The following is a brief introduction to our work. A more detailed report (in Chinese) can be found [here](https://github.com/Andy19961017/ML2018SPRING/blob/master/final/Report.pdf).

## Task Description
Given the transcript of a conversation and 5 sentence candidates to choose from, the model can choose the sentence that is most likely to appear right next to the conversation.  

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
The answer should be 3. Note that the real problems in this task are in Chinese instead of English.
In the following passage, we refer to the given conversation as ***context*** and the correct answer (i.e., the sentence following the context) as the ***following sentence***.

## Key Results
1.Accuracy: ~53.5% (without ensemble), ~57.8% (ensembled).  
2.Ranked 1st out of 11 teams in the Kaggle competition. [Link](https://www.kaggle.com/c/ml-2018spring-final-tv-conversation/leaderboard). Team name: NTU_b04901067_燃燒肝和GPU

## Methodology
### 1. Data processing
Training data is a corpus based on a Chinese novel. We need to generate positive and negative training samples based on the corpus. A positive sample is formed in [A, B] where A is several consecutive sentences in the corpus (the ***context***) and B is the sentence right after A in the corpus (the ***following sentence***). A negative sample is [A, B'] where B' is a random sentence from the corpus that is not right after A.
More details on data processing are elaborated in the ***Technical Challenge and Solutions*** section.

### 2. Model architecture and training
#### 2a. Word-to-vector
We first train a word-to-vector model based on the corpus, using the [skip gram algorithm](https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c), implemented by [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) package.

#### 2b. Prediction model
The prediction model is a binary classification model. It reads a pair of text sequences and gives a score indicating the likelihood that this pair is a correct ***context - following sentence*** pair.

The prediction model is simply an LSTM text sequence encoder (illustrated in the picture below). We first turn the two input text sequences into two sequences of word vectors. Then, use the sentence encoder to encode the two sequences into two vectors. Lastly, use the inner product (dot operation) of the two vectors as the loss function. This is equivalent to calculating the cosine similarity of the two encoded text sequences. 

<img width="383" alt="model architecture" src="https://user-images.githubusercontent.com/37168711/125164116-2aed5000-e1c3-11eb-8495-dd22b3ec889d.png">

During training, label 1 is given for positive samples and 0 is given for negative samples. The intuition is that the context and following sentence should be about a similar topic. Thus, the cosine similarity of the two should be higher.

Note that this is a completely symmetric model. That is, swapping the context and the following sentence will give the exact same prediction score. This does not make sense since, in many cases, the following sentence only makes sense when put ***after*** the context instead of before it. Therefore, we designed another model architecture with an additional fully connected natural network layer after each encoded sentence (illustrated in the picture below). Using this model, we achieved higher validation accuracy on the training data set, meaning that the model learned to distinguish the context and the following sentence. However, the accuracy of the testing dataset is in fact lower. This is due to the heterogeneity between training and testing data set (training corpus is based on a classic novel and testing data is based on conversations in modern TV shows). The model's fitting capability is too strong that it overfitted on the training data.

<img width="377" alt="model architecture 2" src="https://user-images.githubusercontent.com/37168711/125165400-4bb8a400-e1c9-11eb-9bb4-872d2da3d5ed.png">

### 3. Testing
Each testing data consists of one context and 5 sentences as the potential following sentence. We parse it into 5 ***context - following sentence*** pairs. We use our model to evaluate the likelihood of each pair and choose the one with the highest likelihood as the answer.

## Technical Challenge and Solutions
The major technical of this task is the heterogeneity between the training and testing dataset. As mentioned previously in ***2. Model architecture and training*** under the ***Methodology*** section, the training data is based on a classic novel while the testing data is based on conversations in modern TV shows, which is very different topic and word usage. The challenge can be broken down into two parts: (1) difference in topic and (2) difference in text sequence length.
### 1. Difference in topic
There are two ways to deal with this challenge. First, add regulations to the model to prevent overfitting. As discussed in ***2. Model architecture and training*** under the ***Methodology*** section, we designed the model to be completely symmetric. Using an asymmetric model with better fitting capability results in higher accuracy on the training dataset but lower accuracy on the testing dataset. The second way is to fine-tune the model parameters based on testing data before testing. This results in a ~6% increase in testing accuracy based on experiments. However, this method is prohibited in the competition so is not adopted.
### 2. Difference in text sequence length
Note that, in the testing data, the context is typically a dialogue consists of several sentences instead of one. When parsing the training data (contents of a novel) into training data, we need to decide how many sentences should be concatenated together to form the context of each training sample. An easy way to do so (most teams in the competition did this) is to measure the average text length of the context in the testing data and parse the training data to match this length. However, we designed a clever way to parse the data and yields a ***6% increase in accuracy***. We designed a randomized parser to parse the corpus. The parser is based on some conditional probability rules based on the current sentence length. We tuned the rules of the parser so that the distribution of the text sequence length of the training data is similar to that of the testing data (illustrated in the picture below). In addition, since the parser is randomized, the training dataset parsed each time is not identical. We therefore can use non-identical datasets to train multiple models. Since these models are trained by different datasets, the performance increase when ensembling these models is larger compared with training them with the same dataset.
<img width="758" alt="distribution" src="https://user-images.githubusercontent.com/37168711/125167713-7825ed80-e1d4-11eb-98b8-1920943cf365.png">

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
