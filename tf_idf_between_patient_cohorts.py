from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import re
import sys
import scipy
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

CRISIS_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_CRISIS.csv")
CRISIS_PATIENTS.insert(19,"dataset","crisis")
BROAD_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_BROAD.csv")
BROAD_PATIENTS.insert(19,"dataset","broad")
full_dataframe = pd.concat([BROAD_PATIENTS,CRISIS_PATIENTS])
full_corpus = ','.join(full_dataframe['TEXT'])

crisis_corpus = ','.join(CRISIS_PATIENTS['TEXT'])
broad_corpus = ','.join(BROAD_PATIENTS['TEXT'])

corpus_strings = [crisis_corpus,broad_corpus]

# now lets say we want to do our homework re TF-IDF!
#first, we want to tokenize -- pull out all the words

word_tokens = word_tokenize(full_corpus)  #grab all of the words out of our lines object
stop_words = stopwords.words('english')
tokens_without_sw = [word for word in word_tokens if not word in stop_words]
print(tokens_without_sw)
punct = list(string.punctuation)
tokens_wo_punct = [word for word in tokens_without_sw if not word in punct]
print(tokens_wo_punct)
punct_extra = list('”')
tokens_wo_punct2 = [word for word in tokens_wo_punct if not word in punct_extra]
print(tokens_wo_punct2)
final_tokens = tokens_wo_punct2
len(word_tokens)
len(final_tokens)
final_string=(" ").join(final_tokens)
wordcloud = WordCloud(width = 1000, height = 500).generate(final_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("SC Python Wordcloud"+".png", bbox_inches='tight')
plt.show()
plt.close()

### TF-IDF

#import the TfidfVectorizer from Scikit-Learn.
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
transformed_documents = vectorizer.fit_transform(corpus_strings)

transformed_documents_as_array = transformed_documents.toarray()
# use this line of code to verify that the numpy array represents the same number of documents that we have in the file list
len(transformed_documents_as_array)
# make the output folder if it doesn't already exist
# Path("./tf_idf_output").mkdir(parents=True, exist_ok=True)

# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)

    # output to a csv using the enumerated value for the filename
    one_doc_as_df.to_csv("tf_idf.csv")
