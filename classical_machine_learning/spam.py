import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from imblearn.over_sampling import SMOTE


NUM_CV = 5 # 5-fold cross validation

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


def vectorization(data):
    """
    TFIDF vectorizer.
    """
    vectorizer = TfidfVectorizer()
    ##### (Feature engineering) remove English stop words; include 1-grams and 2-grams
    # vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(data["X_train"])
    data["X_train_vec"] = vectorizer.transform(data["X_train"])
    data["X_test_vec"] = vectorizer.transform(data["X_test"])
    print("\nFirst few feature names:")
    print(vectorizer.get_feature_names()[0:10]) # Print first ten feature names  
    print ("\nShape of sparse matrix: ", data["X_train_vec"].shape)
    print ("Amount of non-zero occurences: ", data["X_train_vec"].nnz)
    print ("Sparsity: %.2f%%" % (100.0 * data["X_train_vec"].nnz \
            / (data["X_train_vec"].shape[0] * data["X_train_vec"].shape[1])))
    print("\nTraining label count:")
    print(np.unique(data["y_train"], return_counts=True))
    return vectorizer, data


def smote(data):
    """
    (Modeling technique) SMOTE.
    Balance training/test set samples.
    """
    sm = SMOTE(random_state=100)
    data["X_train_vec"], data["y_train"] = sm.fit_sample(data["X_train_vec"], data["y_train"])
    print("\nAfter SMOTE, training label count:")
    print(np.unique(data["y_train"], return_counts=True))


def fit_model(models, data):
    """
    Fit machine learning models, given model parameters.
    """
    cv_score = {} # store area under ROC score from cross-validation
    auc = {} # store area under ROC curve score
    predict_prob = {} # store predictions of class 1, i.e. spam
    for name, model in models.items():  
        # CV to check algorithm robustness
        cv_score[name] = cross_val_score(model, data["X_train_vec"], data["y_train"], cv=NUM_CV, scoring='roc_auc')
        model.fit(data["X_train_vec"], data["y_train"])
        predict_prob[name + "_train"] = pd.DataFrame(model.predict_proba(data["X_train_vec"]))[1]
        predict_prob[name + "_test"] = pd.DataFrame(model.predict_proba(data["X_test_vec"]))[1]
        auc[name + '_train'] = roc_auc_score(data["y_train"], predict_prob[name + "_train"])
        auc[name + '_test'] = roc_auc_score(data["y_test"], predict_prob[name + "_test"])
    return cv_score, auc, predict_prob


def train_model(data):
    """
    Train machine learning models.
    """
    classifer_names = ["LogisticRegression", "RandomForestClassifier"]
    models = {
        "LogisticRegression" : LogisticRegression(),
        "RandomForestClassifier" : RandomForestClassifier()
        }
    ##### (Modeling technique) CV parameter tuning
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 100]}
    lr = GridSearchCV(LogisticRegression(), param_grid, scoring="roc_auc", cv=NUM_CV)
    lr.fit(data["X_train_vec"], data["y_train"])
    print("\nBest parameter in Logistic Regression. C = %f" % (lr.best_params_["C"]))
    models["LogisticRegression"] = LogisticRegression(C=lr.best_params_["C"]) # Replace existing model
    param_grid = { 
        "n_estimators" : [20, 40, 60, 80, 100],
        "max_depth" : [5, 10, 15, 20]
    }
    rfc = GridSearchCV(RandomForestClassifier(), param_grid, scoring="roc_auc", cv=NUM_CV)
    rfc.fit(data["X_train_vec"], data["y_train"])
    print("\nBest parameter in Random Forest. n_estimators = %d. max_depth = %d" % \
            (rfc.best_params_["n_estimators"], rfc.best_params_["max_depth"]))
    models["RandomForestClassifier"] = RandomForestClassifier(
            n_estimators=rfc.best_params_["n_estimators"],
            max_depth=rfc.best_params_["max_depth"]) # Replace existing model
    cv_score, auc, predict_prob = fit_model(models, data)
    return models, cv_score, auc, predict_prob


def print_ml_performance(models, cv_score, auc):
    """
    Print model performance
    """
    for name in models.keys():
        print("\nModel performance:")
        print("%s, %s-fold Cross Validation - auc score mean: %.4f, std: %.4f" \
                % (name, NUM_CV, cv_score[name].mean(), cv_score[name].std()))
        print("%s, Training set - auc score: %.4f" % (name, auc[name + "_train"]))
        print("%s, Test set - auc score: %.4f" % (name, auc[name + "_test"]))


def understand_misclassification(name, predict_prob, data):
    """
    Understand misclassified sms text by the model.
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


def feature_importance(vectorizer, models):
    """
    Print top few important features.
    """
    feature_importance = pd.DataFrame(
        {"token": vectorizer.get_feature_names(),
        "RandomForestClassifier": models["RandomForestClassifier"].feature_importances_,
        "LogisticRegression": models["LogisticRegression"].coef_[0],
    })
    print("\nTop influencing features sorted by feature importance score:")
    for name, model in models.items():
        print(feature_importance.sort_values(name, ascending=False)[[name, "token"]].head())


def save_model(models):
    """
    Save model to disk.
    """
    print("\nSave model to disk")
    for name, model in models.items():
        joblib_filename = name + ".pkl"  
        joblib.dump(model, joblib_filename)


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
vectorizer, data = vectorization(data)

# SMOTE
smote(data)

# Train machine learning models
models, cv_score, auc, predict_prob = train_model(data)

# Print model performance
print_ml_performance(models, cv_score, auc)

# Understand misclassification
understand_misclassification("LogisticRegression", predict_prob, data) # e.g. LogisticRegression

# Feature importance
feature_importance(vectorizer, models)

# Save model to disk
# save_model(models)
