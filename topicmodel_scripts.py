import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import os
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import re

my_files = pd.read_csv('full_dataframe_sentenced.csv')

trimmed = my_files['Sentence'].head(100)

quoted_full_text = ','.join(trimmed)
my_files = []
doc=[]
word_tokens = word_tokenize(quoted_full_text.strip())
            word_tokens = [w.lower() for w in word_tokens]
#            word_tokens = [ps.stem(w) for w in word_tokens]
            stop_words = set(stopwords.words('english'))
            filtered_sentence = [w for w in word_tokens if not w in stop_words and w not in string.punctuation]
            doc.extend(filtered_sentence)
    my_files.append(doc)

vectorizer = CountVectorizer()
data_vectorized = vectorizer.fit_transform(doc).toarray()


# Build a basic LDA Model (basic)
lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1                # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)  # Model attributes

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())

#### next step ######
## to get the statistically optimum model using a grid search
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}


lda_model = LatentDirichletAllocation(batch_size=128,
                                      doc_topic_prior=None,
                                      evaluate_every=-1,
                                      learning_method='online',
                                      learning_offset=10.0,
                                      max_doc_update_iter=100,
                                      max_iter=10,
                                      mean_change_tol=0.001,
                                      n_jobs=-1,
                                      perp_tol=0.1,
                                      random_state=100,
                                      topic_word_prior=None,
                                      verbose=1)
model = GridSearchCV(lda_model, param_grid=search_params)


# Do the Grid Search
model.fit(data_vectorized)

# compare Models
best_lda_model = model.best_estimator_
# Print out model Parameters
print("Optimal Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Optimal Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Optimal Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
# See model parameters
print(best_lda_model.get_params())


# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


#use the show topics function to look at the topics for each keyword
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords.to_csv('topics.csv')