# This is a BoW classifier to detect the presence of provider scare quotes in provider clinical notes

import pandas as pd
import numpy
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
# Import modules for evaluation purposes
# Import libraries for predcton
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc,f1_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from wordcloud import WordCloud
# Load in Data


gold_standard = pd.read_csv("gold_standard_bias_annotation_doc_training.csv")

# SPLIT
# 1. First, for HW 7, write a script which loads your labeled dataset and divides it into a training-
# test split. Consider the various methods we discussed in class, and think about how you want
# to approach – e.g. using k-fold within your training set as your validation step. If you decide
# not to include a validation step, note why in your write up. Push this script to github. Write a
# short paragraph in overleaf re: the choices you made and why you made them and submit your
# paragraph on canvas (can be included in the same doc as hw 8 below).
#BOW MODEL
# 2. Next, for HW 8, write a script that trains a BOW model – could be any non-neural model you
# like. Consider at least two variation of the features of your model—e.g. using counts v. TF-IDF
# representations of your text—rather than two model types. Write up a paragraph describing the
# choices you made in your overleaf doc. Consider the many available tutorials for this sort of thing
# (e.g. here and the sklearn package). Finally, estimate the F1 score and plot a precision/recall
# curve with only one model specification plotted. Consider using a tutorial (e.g. here). Include the
# plot in your write up on overleaf and submit on Canvas.

#X_train, X_test, y_train, y_test = train_test_split(gold_standard["Sentence"], gold_standard["quote_use"].values , test_size=0.20, random_state=0)
# Use k-fold
# Show the size of our datasets
#print('X Train Size:',X_train.shape)
# print('X Test Size:',X_test.shape)
X = gold_standard['Sentence']
y = gold_standard['quote_use']
#Hyperparameters
max_features_model = 3000
splits = 5
model_name = "MultinomialNaiveBayes"
vectorization = "count"
#Splitting and model
skf = StratifiedKFold(n_splits=splits)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    vect = CountVectorizer(ngram_range=(1,3), max_features=max_features_model , stop_words="english",token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)

print(metrics.classification_report(y_test, y_pred_class, digits=5))
class_metrics_files = metrics.classification_report(y_test, y_pred_class, digits=5)
text_file = open(str(vectorization)+ "metrics_classification_report_" + str(model_name) + str(max_features_model) +" _features_"+ str(splits)+"_splits"+".txt", "w")
n = text_file.write(class_metrics_files)
text_file.close()


# try multiple ways of calculating features
# Create the numericalizer TFIDF for lowercase
# tfidf = TfidfTransformer(encoding = "utf-8")
# Numericalize the train dataset
# tf_idf_train = tfidf.fit_transform(X_train.values.astype('U'))
# Numericalize the test dataset
# tf_idf_test = tfidf.transform(X_test.values.astype('U'))

# pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),('tfid', TfidfTransformer())]).fit(X_train)

# Create the confussion matrix
def plot_confussion_matrix(y_test, y_pred_class):
    ''' Plot the confussion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred_class)

    # Create a dataframe with the confussion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                  range(cm.shape[1]))
    plt.figure(figsize = (10,7))
    # Plot the confussion matrix
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, annot=True,fmt='.0f',annot_kws={"size": 10})# font size
    cf_plot_name = str(vectorization) + "confusion_matrix" + str(model_name) + str(max_features_model)+ "features_"+ str(splits)+ "_splits"+ ".png"
    plt.savefig(cf_plot_name, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_test, y_pred_class):
    ''' Plot the ROC curve for the target labels and predictions'''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_class, pos_label=1)
    roc_auc= auc(fpr,tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    cf_plot_name = str(vectorization) + "ROC Curve" + str(max_features_model) + "features.png"
    plt.savefig(cf_plot_name, bbox_inches='tight')
    plt.show()


plot_confussion_matrix(y_test, y_pred_class)
plot_roc_curve(y_test, y_pred_class)

