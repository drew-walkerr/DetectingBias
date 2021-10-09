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

#sentences = []
#for row in full_dataframe['TEXT'].iteritems():
#    for sentence in row[1].split('.'):
#        if sentence != '':
 #           sentences.append((row[0], sentence))
#new_df = pd.DataFrame(sentences, columns=['ROW_ID_x', 'SENTENCE'])

nlp.add_pipe('sentencizer')
full_dataframe["Sentence"] = full_dataframe["TEXT"].apply(lambda x: [sent.text for sent in nlp(x).sents])
full_dataframe = full_dataframe.explode("Sentence", ignore_index=True)
full_dataframe.rename(columns={"Unnamed: 0": "ROW_ID_new"}, inplace=True)
full_dataframe.index.name = "Sentence ID"

full_dataframe['Sentence'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)

full_dataframe = full_dataframe[full_dataframe['Sentence'].map(len) > 15]
full_dataframe["questioning_credibility"] = ""
full_dataframe["questioning_credibility_quote"] = ""
full_dataframe["disapproval"] = ""
full_dataframe["disapproval_quote"] = ""
full_dataframe["stereotyping"] = ""
full_dataframe["stereotyping_quote"] = ""
full_dataframe["difficult_patient"] = ""
full_dataframe["difficult_patient_quote"] = ""
full_dataframe["unilateral_decisions"] = ""
full_dataframe["unilateral_decisions_quote"] = ""
full_dataframe["compliment"] = ""
full_dataframe["compliment_quote"] = ""
full_dataframe["approval"] = ""
full_dataframe["approval_quote"] = ""
full_dataframe["self_disclosure"] = ""
full_dataframe["self_disclosure_quote"] = ""
full_dataframe["minimize_blame"] = ""
full_dataframe["minimize_blame_quote"] = ""
full_dataframe["personalize"] = ""
full_dataframe["personalize_quote"] = ""
full_dataframe["bilateral_decision"] = ""
full_dataframe["bilateral_decision_quote"] = ""
full_dataframe["notes_or_concerns"] = ""


full_dataframe.to_csv("full_dataframe_sentenced.csv")


head_dataframe <- full_dataframe.head()

head_dataframe.to_csv("head_dataframe_preprocessed.csv")
