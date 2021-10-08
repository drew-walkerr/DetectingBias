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
import spacy
from spacy.lang.en import English

nlp = English()  # just the language with no model

CRISIS_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_CRISIS.csv")
CRISIS_PATIENTS.insert(19,"dataset","crisis")
BROAD_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_BROAD.csv")
BROAD_PATIENTS.insert(19,"dataset","broad")
full_dataframe = pd.concat([BROAD_PATIENTS,CRISIS_PATIENTS])
full_corpus = ','.join(full_dataframe['TEXT'])

crisis_corpus = ','.join(CRISIS_PATIENTS['TEXT'])
broad_corpus = ','.join(BROAD_PATIENTS['TEXT'])

corpus_strings = [crisis_corpus,broad_corpus]


sentences = []
for row in full_dataframe.iterrows():
    for sentence in sent_tokenize(full_dataframe['TEXT']):
        sentences.append((row[1], sentence))
new_df = pandas.DataFrame(sentences, columns=['ROW_ID_x', 'SENTENCE'])

#sentences = []
#for row in full_dataframe['TEXT'].iteritems():
#    for sentence in row[1].split('.'):
#        if sentence != '':
 #           sentences.append((row[0], sentence))
#new_df = pd.DataFrame(sentences, columns=['ROW_ID_x', 'SENTENCE'])

full_dataframe["Sentence"] = full_dataframe["TEXT"].apply(lambda x: sent_tokenize(x))

sentence_tokens = sent_tokenize(full_dataframe['TEXT'])
nlp.add_pipe('sentencizer')
full_dataframe["Sentence"] = full_dataframe["TEXT"].apply(lambda x: [sent.text for sent in nlp(x).sents])
full_dataframe = full_dataframe.explode("Sentence", ignore_index=True)
full_dataframe.rename(columns={"Unnamed: 0": "ROW_ID_new"}, inplace=True)
full_dataframe.index.name = "Sentence ID"


full_dataframe.to_csv("full_dataframe_sentenced.csv")
