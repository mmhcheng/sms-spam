# Deep Learning
Demostrates how Deep Learning (Recursive Neural Network) models could be trained to classify sms (text) spam, through a Python script [spam-rnn.py](spam-rnn.py) that uses [Keras](https://keras.io/). 


## 1. Understand raw data
Dataset can be found in [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

First few lines of data:
```
     v1                                                 v2 Unnamed: 2  \
0   ham  Go until jurong point, crazy.. Available only ...        NaN   
1   ham                      Ok lar... Joking wif u oni...        NaN   
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   
3   ham  U dun say so early hor... U c already then say...        NaN   
4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   

  Unnamed: 3 Unnamed: 4  
0        NaN        NaN  
1        NaN        NaN  
2        NaN        NaN  
3        NaN        NaN  
4        NaN        NaN 
```


## 2. Data pre-processing
- Drop unnecessary columns
- Rename columns
- Convert target labels into numerical

First few lines of data after pre-processing:
```
  label                                                sms  label_numerical
0   ham  Go until jurong point, crazy.. Available only ...                0
1   ham                      Ok lar... Joking wif u oni...                0
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...                1
3   ham  U dun say so early hor... U c already then say...                0
4   ham  Nah I don't think he goes to usf, he lives aro...                0
```


## 3. Training / Test set splitting
We split data into 80% as training set and 20% as test set.
```
Training set contains 3855 hams (86.5%) and 602 spams (13.5%).
Test set contains 970 hams (87%) and 145 spams (13%).
```


## 4. Tokenize input text
Keras's [Tokenizer](https://keras.io/preprocessing/text/#tokenizer) is used to vectorize text into sequences of word indexes. For demostration, list of words are restricted to the top 1000 most common words, max text length = 20, post padding and post truncating.

First few word index mapping:
```
'i': 1, 'to': 2, 'you': 3, 'a': 4, 'the': 5
```

## 5. RNN model training
[Keras](https://keras.io/) library is adopted in the model training. For demostration, word embedding of output dimension = 32 is used to map input sequences into relative representation of vectors. More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding).

One layer of Long-Short Term Memory (**LSTM**) layer with 24 units is used to pass local information to later position in the sequence. More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory).

Finally, a densely-connected NN layer of 1 unit with sigmoid activation function is used to classify spam on the given sms text.

10 epochs are used to train the model.

Model performance:
```
RNN, Training set - auc score: 0.9965
RNN, Test set - auc score: 0.9936
```
Area under ROC curve score by this Deep RNN Learning model is better than the performance achieved by some classical machine learning algorithms, e.g. Random Forest or Logistic Regression in [spam.py](https://github.com/mmhcheng/sms-spam). **Note**: model performance on the training set is slightly better than the test set, which indicates the model is slightly overfitted to the training set. Regularization, e.g. Dropout, can be used to resolve such issue.

e.g. If we assume threshold = 0.5, i.e. prediction is "spam" if output probability >= 0.5, else prediction is "ham". **Note:** In practice, actual threshold is picked based on appropriate metric, e.g. cost-benefit.

Confusion Matrix:
Column: truth ("ham", "spam")
Row: prediction ("ham", "spam")
```
[[960  10]
 [ 10 135]]
 ```
 
 Precision/Recall:
 ```
             precision    recall  f1-score   support

          0     0.9897    0.9897    0.9897       970
          1     0.9310    0.9310    0.9310       145

avg / total     0.9821    0.9821    0.9821      1115
```

False positives (ham incorrectly classified as spam):
```
1248    HI HUN! IM NOT COMIN 2NITE-TELL EVERY1 IM SORR...
4633          These won't do. Have to move on to morphine
5142    Now that you have started dont stop. Just pray...
3413                              No pic. Please re-send.
2339    Cheers for the message Zogtorius. IåÕve been s...
4727    I (Career Tel) have added u as a contact on IN...
1219    True. It is passable. And if you get a high sc...
1383    Please reserve ticket on saturday eve from che...
1289    Hey...Great deal...Farm tour 9am to 5pm $95/pa...
1418                  Lmao. Take a pic and send it to me.
```

False negatives (spam incorrectly classified as ham):
```
2429    Guess who am I?This is the first time I create...
5540    ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...
4947    Hi this is Amy, we will be sending you a free ...
1136    Dont forget you can place as many FREE Request...
671            SMS. ac sun0819 posts HELLO:\You seem cool
610     22 days to kick off! For Euro2004 U will be ke...
750     Do you realize that in about 40 years, we'll h...
2773    How come it takes so little time for a child w...
868     Hello. We need some posh birds and chaps to us...
2913    Sorry! U can not unsubscribe yet. THE MOB offe...
```


=============================================================

# Potential improvement
- Number of most common top word in Tokenizer
- Max number of text length in Tokenizer
- Dimension of Word Embedding
- Number of units in LSTM
- More number of inner NN layers
- Number of epochs used to train the model
- Incorporate Dropout layers to avoid overfitting

