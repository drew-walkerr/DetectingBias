import pandas as pd
import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

crisis_df_raw = pd.read_csv("crisis_dataframe_preprocessed.csv")
# Regex grab quotes '\"(.+?)\"'

sentences_all = ','.join(crisis_df_raw["Sentence"])


quoted_words = re.findall('\"(.+?)\"',sentences_all)
#This version actually has some different non-quotes in here that are really long
print(quoted_words)
# Extract quoted words

quoted_full_text = ','.join(quoted_words)
# unnest tokens of quoted words by word, bigram,trigram
quoted_tokens = wordpunct_tokenize(quoted_full_text)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black',max_words=300).generate(quoted_full_text)
plt.figure(figsize=(15,10))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# this really is dominated by large charts that were accidentally captured here.
# R's str_extract may be manipulating how the regex works differently, or perhaps an issue with whitespace.
# Now we have quoted df, overall df
# From overall, get total number of charts, Sentences
# From quotes, get same
# Merge into one table



### SUBJECTIVITY
# Tokenize full df
full_tokens = pd.DataFrame(wordpunct_tokenize(sentences_all)).rename(columns={0:'word'})

# read in Subjectivity mpqa "subjectivity.csv.gz"

subjectivity = pd.read_csv('subjectivity.csv.gz', compression='gzip',
    header=None, sep=',', quotechar='"')
# rename headings word=V1, subjectivity=V2, sentiment=V3
subjectivity_tidy = subjectivity.rename(columns = {0: 'word', 1: 'subjectivity',2:'sentiment'}, inplace = False)
# inner join tokenized full df with mpqa words


subjective_corpus = pd.merge(full_tokens,subjectivity_tidy,on='word')
# count for each word
counts = subjective_corpus.word.value_counts().reset_index()
#already did distinct
counts2 = counts.rename(columns = {'index':'word','word':'count'},inplace=False)

subjective_corpus_counts = pd.merge(counts2,subjective_corpus,on="word")
# group by, distinct(word,subjectivity), slice_max(5)
#
