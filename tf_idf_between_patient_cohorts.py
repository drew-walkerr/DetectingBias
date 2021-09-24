### to use a TF-IDF hack
# Load in data
import pandas as pd

CRISIS_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_CRISIS.csv")
BROAD_PATIENTS = pd.read_csv("SC_NOTES_PATIENTS_BROAD.csv")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1), max_df = .6, min_df = .01)
X = vectorizer.fit_transform(my_files)
feature_names = vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.head()
data = df.transpose()