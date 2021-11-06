import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import os
import spacy
from spacy import displacy
import matplotlib.pyplot as plt

crisis_df_raw = pd.read_csv("crisis_dataframe_preprocessed.csv")
# Regex grab quotes '\"(.+?)\"'

sentences_all = ','.join(crisis_df_raw["Sentence"])

pt_words = re.findall(r"([^.]*?patient[^.]*\.)",sentences_all)

quoted_words = re.findall('\"(.+?)\"',sentences_all)
# NER on patient_words

sentences_all = ','.join(pt_words)

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentences_all)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

sentence_spans = list(doc.sents)

docs = list(nlp.pipe(pt_words))
#displacy.serve(sentence_spans, style="dep")

def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
    """Extract tokens and metadata from individual spaCy doc."""
    return [
        (i.text, i.i, i.lemma_, i.ent_type_, i.tag_,
         i.dep_, i.pos_, i.is_stop, i.is_alpha,
         i.is_digit, i.is_punct) for i in doc
    ]

def tidy_tokens(docs):
    """Extract tokens and metadata from list of spaCy docs."""

    cols = [
        "doc_id", "token", "token_order", "lemma",
        "ent_type", "tag", "dep", "pos", "is_stop",
        "is_alpha", "is_digit", "is_punct"
    ]

    meta_df = []
    for ix, doc in enumerate(docs):
        meta = extract_tokens_plus_meta(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols[1:]
        meta = meta.assign(doc_id=ix).loc[:, cols]
        meta_df.append(meta)

    return pd.concat(meta_df)

dep_df = tidy_tokens(docs)

dep_df.to_csv("dep_df.csv")

