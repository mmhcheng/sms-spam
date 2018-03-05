import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def preprocessing(df):
    """
    Drop unnecessary columns.
    Rename columns.
    Convert target labels into numerical.
    """
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1":"label", "v2":"sms"})
    # Convert label into numerical
    df["label_numerical"] = df.label.map({"ham":0, "spam":1})
    return df


def tokenize(data, num_top_word, max_text_length):
    """
    Tokenize input string of characters.
    """
    tokenizer = Tokenizer(num_words=num_top_word)
    tokenizer.fit_on_texts(data["X_train"])
    train_sequences = tokenizer.texts_to_sequences(data["X_train"])
    data["X_train_seq"] = pad_sequences(train_sequences,
                                              maxlen=max_text_length,
                                              padding='post',
                                              truncating='post')
    train_sequences = tokenizer.texts_to_sequences(data["X_test"])
    data["X_test_seq"] = pad_sequences(train_sequences,
                                             maxlen=max_text_length,
                                             padding='post',
                                             truncating='post')
    print("\nTraining label count:")
    print(np.unique(data["y_train"], return_counts=True))


def smote(data):
    """
    (Modeling technique) SMOTE.
    Balance training/test set samples.
    """
    sm = SMOTE(random_state=100)
    data["X_train_seq"], data["y_train"] = sm.fit_sample(data["X_train_seq"], data["y_train"])
    print("\nAfter SMOTE, training label count:")
    print(np.unique(data["y_train"], return_counts=True))


def train_rnn(data, num_top_word, embedding_vector_length, max_text_length):
    """
    Train a RNN classification model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=num_top_word,
                        output_dim=embedding_vector_length,
                        input_length=max_text_length))
    model.add(LSTM(24))
    # model.add(Dense(64, activation='relu', input_dim=embedding_vector_length))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    model.fit(x=data["X_train_seq"], y=data["y_train"], epochs=10, validation_split=0.2)
    return model


def compute_performance(models, predict_prob, auc):
    """
    Compute model performance.
    """
    for name, model in models.items():
        predict_prob[name + "_train"] = model.predict_proba(data["X_train_seq"]).flatten()
        predict_prob[name + "_test"] = model.predict_proba(data["X_test_seq"]).flatten()
        auc[name + "_train"] = roc_auc_score(data["y_train"], predict_prob[name + "_train"])
        auc[name + "_test"] = roc_auc_score(data["y_test"], predict_prob[name + "_test"])
        print("\nModel performance:")
        print("%s, Training set - auc score: %.4f" % (name, auc[name + "_train"]))
        print("%s, Test set - auc score: %.4f" % (name, auc[name + "_test"]))


def understand_misclassification(name, predict_prob, data):
    """
    Understand misclassification by the model.
    """
    # Assume threshold
    threshold = 0.5
    print("\ne.g. for %s" % (name))
    prediction = np.where(predict_prob[name + "_test"] > 0.5, 1, 0)
    print("\n Confusion Matrix:")
    print(confusion_matrix(data["y_test"], prediction)) # Confusion matrix
    print(classification_report(data["y_test"], prediction, digits=4))
    # Understand mis-classifications
    print("\nFalse positives (ham incorrectly classified as spam):")
    print(data["X_test"][prediction > data["y_test"]]) # False positives
    print("\nFalse negatives (spam incorrectly classified as ham):")
    print(data["X_test"][prediction < data["y_test"]]) # False negatives


########## Run the code ##########
# Import data
# https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
print("\nFirst few lines of data:")
print(df.head()) # print first few lines

df = preprocessing(df)
print("\nFirst few lines of data after pre-processing:")
print(df.head()) # Print first few lines
# print(df.groupby("label").describe()) # Count observations

# Training / Test split
data = {}
data["X_train"], data["X_test"], data["y_train"], data["y_test"] = train_test_split(
    df["sms"], df["label_numerical"], test_size=0.2, random_state=100)

# Tokenize input
num_top_word = 1000
max_text_length = 20
tokenize(data, num_top_word, max_text_length)

# SMOTE
# smote(data)

# Recursive Neural Network
embedding_vector_length = 32
models = {}
models["RNN"] = train_rnn(data, num_top_word, embedding_vector_length, max_text_length)

# Final evaluation of the model
predict_prob = {}
auc = {}
compute_performance(models, predict_prob, auc)

# Understand misclassification
understand_misclassification("RNN", predict_prob, data)

# Save model
# models["RNN"].save('rnn.h5')
