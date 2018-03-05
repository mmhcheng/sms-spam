# Classical Machine Learning
Demonstrates how machine learning models could be trained to classify sms (text) spam, through a Python script [spam.py](spam.py) that uses [Scikit-learn](http://scikit-learn.org/stable/). 


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


## 4. Vectorize input text
TFIDF vectorizer is used to vectorize text into model features in this classification problem.
It computes the importantance of a word in a document inside a collection. More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

First few vectorized feature names (tokens):
```
['00', '000', '000pes', '008704050406', '0089', '0121', '01223585236', '01223585334', '0125698789', '02']
```

For training set,
```
Shape of sparse matrix:  (4457, 7793)
Amount of non-zero occurences:  59469
Sparsity: 0.17%
```


## 5. Model performance
For demostration, (Vanilla) Logistic Regression and Random Forest are used as baseline to understand the complexity of the problem. _5-fold cross-validation is used to understand model accuracy **robustness**_.

Here, area-under-ROC-curve (AUC) is picked as the performance metric because it is independent of any threshold for the classification problem. **Note:** In practice, we will consider more metrics, e.g. weighted-F1, custom metric based on cost-benefit, etc, according to certain desired threshold.


Model performance:
```
LogisticRegression, 5-fold Cross Validation - auc score mean: 0.9898, std: 0.0038
LogisticRegression, Training set - auc score: 0.9977
LogisticRegression, Test set - auc score: 0.9906

RandomForestClassifier, 5-fold Cross Validation - auc score mean: 0.9819, std: 0.0065
RandomForestClassifier, Training set - auc score: 1.0000
RandomForestClassifier, Test set - auc score: 0.9824
```

e.g. For *LogisticRegression*, if we assume threshold = 0.5, i.e. prediction is "spam" if output probability >= 0.5, else prediction is "ham". **Note:** In practice, actual threshold is picked based on appropriate metric, e.g. cost-benefit.

Confusion Matrix:
Column: truth ("ham", "spam")
Row: prediction ("ham", "spam")
```
[[969   1]
 [ 27 118]]
 ```
 
 Precision/Recall:
 ```
              precision    recall  f1-score   support

          0     0.9729    0.9990    0.9858       970
          1     0.9916    0.8138    0.8939       145

avg / total     0.9753    0.9749    0.9738      1115
```

False positives (ham incorrectly classified as spam):
```
4727    I (Career Tel) have added u as a contact on IN...
```

False negatives (spam incorrectly classified as ham):
```
1171    Got what it takes 2 take part in the WRC Rally...
2429    Guess who am I?This is the first time I create...
5540    ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...
4947    Hi this is Amy, we will be sending you a free ...
3753    Bloomberg -Message center +447797706009 Why wa...
3423    Am new 2 club & dont fink we met yet Will B gr...
1136    Dont forget you can place as many FREE Request...
4309    Someone U know has asked our dating service 2 ...
671            SMS. ac sun0819 posts HELLO:\You seem cool
750     Do you realize that in about 40 years, we'll h...
2773    How come it takes so little time for a child w...
4353    important information 4 orange user 0789xxxxxx...
1639    FreeMsg:Feelin kinda lnly hope u like 2 keep m...
3572    You won't believe it but it's true. It's Incre...
1673    Monthly password for wap. mobsi.com is 391784....
4211    Missed call alert. These numbers called but le...
4256    important information 4 orange user . today is...
5381           You have 1 new message. Call 0207-083-6089
4392    RECPT 1/3. You have ordered a Ringtone. Your o...
2718    18 days to Euro2004 kickoff! U will be kept in...
1268    Can U get 2 phone NOW? I wanna chat 2 set up m...
868     Hello. We need some posh birds and chaps to us...
2574    Your next amazing xxx PICSFREE1 video will be ...
3748    Dear Voucher Holder 2 claim your 1st class air...
54      SMS. ac Sptv: The New Jersey Devils and the De...
2913    Sorry! U can not unsubscribe yet. THE MOB offe...
2679    New Tones This week include: 1)McFly-All Ab..,...
```


## 6. Feature importance
Top influencing features sorted by descending feature importance score:
```
      LogisticRegression token
7127            4.384429   txt
1630            4.100367  call
6813            3.440591  text
3017            3.234678  free
6523            3.195478  stop
```

```
      RandomForestClassifier  token
7127                0.066204    txt
1630                0.037207   call
307                 0.029935   150p
7600                0.019938    won
4843                0.017357  nokia
```
=============================================================

# Potential improvement
## 7. Model engineering - SMOTE
In dealing with imbalanced data, many researches show that oversample the minority class or downsample majority class to get a balanced data set will improve model performance of most machine learning models. Here, SMOTE is used to oversample minority class. More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE).

```
Before SMOTE, Training set contains 3855 hams (86.5%) and 602 spams (13.5%).
After SMOTE, Training set contains 3855 hams (50%) and 3855 spams (50%).
Test set contains 970 hams (87%) and 145 spams (13%).
```

After SMOTE, model performance:
```
LogisticRegression, 5-fold Cross Validation - auc score mean: 0.9991, std: 0.0007
LogisticRegression, Training set - auc score: 0.9997
LogisticRegression, Test set - auc score: 0.9912

RandomForestClassifier, 5-fold Cross Validation - auc score mean: 0.9986, std: 0.0027
RandomForestClassifier, Training set - auc score: 1.0000
RandomForestClassifier, Test set - auc score: 0.9896
```
Cross validation auc score mean (is increased) and std (is reduced) are improved for both Logistic Regression and Random Forest, comparing to the original result.


## 8. Model engineering - Hyperparameter tuning
Another aspect of model engineering is hyperparameter tuning. In addition, model performance on the cross-validation set is slightly better than the test set, which indicates the model is slightly overfitted to the training set. Regularization can help to resolve this problem, e.g. set an appropriate C value in Logistic Regression; num_of_tree or max_depth in Random Forest, etc. As an example, scikit-learn's GridSearchCV is demostrated.

After grid search on hyperparameter tuning, model performance:
```
LogisticRegression, 5-fold Cross Validation - auc score mean: 0.9918, std: 0.0037
LogisticRegression, Training set - auc score: 1.0000
LogisticRegression, Test set - auc score: 0.9924

RandomForestClassifier, 5-fold Cross Validation - auc score mean: 0.9899, std: 0.0042
RandomForestClassifier, Training set - auc score: 0.9955
RandomForestClassifier, Test set - auc score: 0.9916
```
Cross validation auc score mean (is increased) and std (is reduced) are improved for both Logistic Regression and Random Forest, comparing to the original result.

**Note:** learning curve w.r.t. each model parameter can be used to better understand how model performance is affected. More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Learning_curve).


## 9. Feature engineering
Feature engineering by e.g. creating new features, refining existing features based on domain knowledge, or reviewing false positives / false negatives, can help improving model performance. Reviewing some spam texts that our model missed and our feature tokens, one will think that 1-gram token (e.g. free) may not be sufficient to distinguish spam from ham, while 2- (or more) gram token (e.g. free request) could provide more hint to the model. Also, noise in sms text can be reduced by removing english stop words.

After some feature adjustments, model performance:
```
Model performance:
LogisticRegression, 5-fold Cross Validation - auc score mean: 0.9911, std: 0.0040
LogisticRegression, Training set - auc score: 0.9991
LogisticRegression, Test set - auc score: 0.9861

Model performance:
RandomForestClassifier, 5-fold Cross Validation - auc score mean: 0.9798, std: 0.0055
RandomForestClassifier, Training set - auc score: 1.0000
RandomForestClassifier, Test set - auc score: 0.9756
```
Cross validation auc score mean (is increased) and std (is reduced) are improved for Logistic Regression, but not at all for Random Forest, comparing to the original result. To further improve the model performance, one may want to try other features, e.g. length of sms text.


## 10. More training data
Similar to hyperparameter tuning, learning curve w.r.t. number of training data can be used to understand how model performance is affected. If model performance starts to plateau with increasing number of training data, classical machine learning models may not be sufficient, and one may want to explore deep learning algorithms.
