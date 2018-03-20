import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def preprocessing(df):
    '''
    Drop unnecessary columns.
    Rename columns.
    Convert target labels into numerical.
    '''
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={'v1': 'label', 'v2': 'sms'})
    # Convert label into numerical
    df['label_numerical'] = df.label.map({'ham': 0, 'spam': 1})
    return df


def split_train_test(df):
    '''
    Split training and test.
    '''
    data = {}
    data['X_train'], data['X_test'], data['y_train'], data['y_test'] = train_test_split(
        df['sms'], df['label_numerical'], test_size=0.2, random_state=100)
    return data


def tokenize(data, num_top_word, max_text_length):
    '''
    Tokenize input string of characters.
    '''
    tokenizer = Tokenizer(num_words=num_top_word)
    tokenizer.fit_on_texts(data['X_train'])
    train_sequences = tokenizer.texts_to_sequences(data['X_train'])
    data['X_train_seq'] = pad_sequences(train_sequences,
                                        maxlen=max_text_length,
                                        padding='post',
                                        truncating='post')
    train_sequences = tokenizer.texts_to_sequences(data['X_test'])
    data['X_test_seq'] = pad_sequences(train_sequences,
                                       maxlen=max_text_length,
                                       padding='post',
                                       truncating='post')
    print('\nTraining label count:')
    print(np.unique(data['y_train'], return_counts=True))


def smote(data):
    '''
    (Modeling technique) SMOTE.
    Balance training/test set samples.
    '''
    sm = SMOTE(random_state=100)
    data['X_train_seq'], data['y_train'] = sm.fit_sample(data['X_train_seq'], data['y_train'])
    print('\nAfter SMOTE, training label count:')
    print(np.unique(data['y_train'], return_counts=True))


def train_rnn(data, num_top_word, embedding_vector_length, max_text_length):
    '''
    Train a RNN classification model.
    '''
    model = Sequential()
    model.add(Embedding(input_dim=num_top_word,
                        output_dim=embedding_vector_length,
                        input_length=max_text_length))
    model.add(LSTM(24))
    # model.add(Dense(64, activation='relu', input_dim=embedding_vector_length))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x=data['X_train_seq'], y=data['y_train'], epochs=10, validation_split=0.2)
    return model


def compute_performance(models):
    '''
    Compute model performance.
    '''
    result = {}
    for name, model in models.items():
        result[name] = {}
        result[name]['predict_prob_train'] = model.predict_proba(data['X_train_seq']).flatten()
        result[name]['predict_prob_test'] = model.predict_proba(data['X_test_seq']).flatten()
        result[name]['auc_train'] = roc_auc_score(data['y_train'], result[name]['predict_prob_train'])
        result[name]['auc_test'] = roc_auc_score(data['y_test'], result[name]['predict_prob_test'])
        result[name]['fpr'], result[name]['tpr'], result[name]['roc_threshold'] = \
            roc_curve(data['y_test'], result[name]['predict_prob_test'])
        result[name]['precision'], result[name]['recall'], result[name]['pr_threshold'] = \
            precision_recall_curve(data['y_test'], result[name]['predict_prob_test'])
    return result


def print_ml_performance(models, result):
    '''
    Print model performance
    '''
    for name in models.keys():
        print('\nModel performance:')
        print('%s, Training set - auc score: %.4f' % (name, result[name]['auc_train']))
        print('%s, Test set - auc score: %.4f' % (name, result[name]['auc_test']))


def plot_roc(result):
    plt.figure(0).clf()
    for name in result.keys():
        fpr = result[name]['fpr']
        tpr = result[name]['tpr']
        plt.plot(fpr, tpr, label=(name + ' auc=' + '%.2f' % (result[name]['auc_test'])))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


def plot_pr(result):
    plt.figure(1).clf()
    for name in result.keys():
        precision = result[name]['precision']
        recall = result[name]['recall']
        plt.plot(recall, precision, label=(name))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc='lower right')
    plt.show()


def understand_misclassification(name, result, data, threshold):
    '''
    Understand misclassified sms text by the model.
    '''
    print('\ne.g. for %s' % (name))
    prediction = np.where(result[name]['predict_prob_test'] > threshold, 1, 0)
    print('\n Confusion Matrix:')
    print(confusion_matrix(data['y_test'], prediction))  # Confusion matrix
    print(classification_report(data['y_test'], prediction, digits=4))
    # Understand mis-classifications
    print('\nFalse positives (ham incorrectly classified as spam):')
    print(data['X_test'][prediction > data['y_test']])  # False positives
    print('\nFalse negatives (spam incorrectly classified as ham):')
    print(data['X_test'][prediction < data['y_test']])  # False negatives


########## Run the code ##########
# Import data
# https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
print('\nFirst few lines of data:')
print(df.head())  # print first few lines

df = preprocessing(df)
print('\nFirst few lines of data after pre-processing:')
print(df.head())  # Print first few lines
# print(df.groupby('label').describe()) # Count observations

# Training / Test split
data = split_train_test(df)

# Tokenize input
num_top_word = 1000
max_text_length = 20
tokenize(data, num_top_word, max_text_length)

# SMOTE
# smote(data)

# Recursive Neural Network
embedding_vector_length = 32
models = {}
models['RNN'] = train_rnn(data, num_top_word, embedding_vector_length, max_text_length)

# Final evaluation of the model
result = compute_performance(models)
print_ml_performance(models, result)
plot_roc(result)
plot_pr(result)

# Understand misclassification
name = 'RNN'
threshold = 0.5
understand_misclassification(name, result, data, threshold)

# Save model
# models['RNN'].save('rnn.h5')
