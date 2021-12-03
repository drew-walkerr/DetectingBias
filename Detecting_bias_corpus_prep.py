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
import numpy as np

ICDs = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_ICDs = ICDs.head()
print(peek_ICDs)
ICDs.info()
#Filter for ICD codes for 282.60-282.69, referring to sickle cell types w/wo crisis
#2824 for thalassemia w + w/o crisis (282.41-282.42)
#SCD: 2826,2824
#Chronic Pain 3382
#Opioid dependencies: 3040,3047 (combo),
#HIV/AIDS ^042$

icds_of_interest = ICDs[ICDs['ICD9_CODE'].str.contains('2826|2824|3040|3047|3382|^042$', na=False)]

print(icds_of_interest.head())

icds_of_interest.info()

patients_unique = icds_of_interest['SUBJECT_ID'].drop_duplicates()

NOTES = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')

biased_notes_corpus = NOTES.merge(patients_unique, on = 'SUBJECT_ID')

PATIENTS = pd.read_csv('PATIENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_patients = PATIENTS.head()
print(peek_patients)

biased_notes_patients_corpus = biased_notes_corpus.merge(PATIENTS, on = 'SUBJECT_ID')

biased_notes_patients_corpus.info()
# Find unique chart types and decide to limit them
    # Remove Radiology, ECG, Respiratory, Echo notes
biased_notes_patients_corpus_filtered = biased_notes_patients_corpus[biased_notes_patients_corpus["CATEGORY"].str.contains("Radiology|ECG|Respiratory|Echo")==False]
biased_notes_patients_corpus_filtered.to_csv("biased_notes_patients_corpus_filtered.csv")
biased_notes_patients_corpus_filtered.head(100).to_csv("biased_notes_patients_corpus_filtered.csv")

full_dataframe = biased_notes_patients_corpus_filtered
# Tokenize by sentence
nlp = English()  # just the language with no model

nlp.add_pipe('sentencizer')
full_dataframe["Sentence"] = full_dataframe["TEXT"].apply(lambda x: [sent.text for sent in nlp(x).sents])
full_dataframe = full_dataframe.explode("Sentence", ignore_index=True)
full_dataframe.rename(columns={"Unnamed: 0": "ROW_ID_new"}, inplace=True)
full_dataframe.index.name = "Sentence ID"

full_dataframe['Sentence'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)

regex = "\"(.+?)\""
quoted_dataframe = full_dataframe.loc[full_dataframe['Sentence'].str.contains(regex)]

quoted_dataframe["scare_quote"] = ""
quoted_dataframe["annotator_comments"] = ""
quoted_dataframe["not_patient_quote"] = ""
# 3842 quoted charts sentences total of 1,510,650 sentences
quoted_dataframe2 = quoted_dataframe.drop_duplicates(subset=['Sentence'])
quoted_dataframe2.to_csv("quoted_dataframe_annotate.csv")

len(np.unique(quoted_dataframe2['CGID']))
# 545 caregivers
len(np.unique(quoted_dataframe2['ROW_ID_x']))
len(np.unique(quoted_dataframe2['SUBJECT_ID']))
len(np.unique(quoted_dataframe2['HADM_ID']))

