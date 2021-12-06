# http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
# This is the general format I'm going to use for the BOW classifier for the detecting bias project.

from sklearn import datasets


from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfTransformer
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


X_train, X_test, y_train, y_test = train_test_split(gold_standard["Sentence"], gold_standard["quote_use"].values , test_size=0.20, random_state=0)
# Use k-fold
# Show the size of our datasets
print('X Train Size:',X_train.shape)
print('X Test Size:',X_test.shape)


# Create a Counter of tokens
count_vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, min_df=2)
# Apply it on the train data to get the vocabulary and the mapping. This vocab and mapping is then applied to the test set.
# Before, we convert to Unicode to avoid issues with CountVectorizer
train = count_vectorizer.fit_transform(X_train.values.astype('U'))
test = count_vectorizer.transform(X_test.values.astype('U'))


print('Train size: ',train.shape)
print('Test size: ',test.shape)
vocab = list(count_vectorizer.vocabulary_.items())
print(vocab[:10])

# try multiple ways of calculating features
# Create the numericalizer TFIDF for lowercase
# tfidf = TfidfTransformer(encoding = "utf-8")
# Numericalize the train dataset
# tf_idf_train = tfidf.fit_transform(X_train.values.astype('U'))
# Numericalize the test dataset
# tf_idf_test = tfidf.transform(X_test.values.astype('U'))

# pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),('tfid', TfidfTransformer())]).fit(X_train)


model = MultinomialNB()
model.fit(train, y_train)
print("train score:", model.score(train, y_train))
print("test score:", model.score(test, y_test))

# Create the confussion matrix
def plot_confussion_matrix(y_test, y_pred):
    ''' Plot the confussion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe with the confussion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                  range(cm.shape[1]))
    #plt.figure(figsize = (10,7))
    # Plot the confussion matrix
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, annot=True,fmt='.0f',annot_kws={"size": 10})# font size
    plt.show()


def plot_roc_curve(y_test, y_pred):
    ''' Plot the ROC curve for the target labels and predictions'''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc= auc(fpr,tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

y_pred = model.predict(test)

print(metrics.classification_report(y_test, y_pred,  digits=5))
plot_confussion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)









#BOW MODEL
# 2. Next, for HW 8, write a script that trains a BOW model – could be any non-neural model you
# like. Consider at least two variation of the features of your model—e.g. using counts v. TF-IDF
# representations of your text—rather than two model types. Write up a paragraph describing the
# choices you made in your overleaf doc. Consider the many available tutorials for this sort of thing
# (e.g. here and the sklearn package). Finally, estimate the F1 score and plot a precision/recall
# curve with only one model specification plotted. Consider using a tutorial (e.g. here). Include the
# plot in your write up on overleaf and submit on Canvas.


clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

print('5-fold cross validation:\n')

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes']

for clf, label in zip([clf1, clf2, clf3], labels):

    scores = model_selection.cross_val_score(clf, train, y_train,
                                              cv=5,
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


# Ensemble voting

from mlxtend.classifier import EnsembleVoteClassifier

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, eclf], labels):

    scores = model_selection.cross_val_score(clf, as.array(train), as.array(y_train),
                                              cv=5,
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))