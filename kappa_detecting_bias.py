import pandas as pd
import sklearn
from sklearn.metrics import cohen_kappa_score

gold_standard = pd.read_csv("gold_standard_bias_annotation_doc_training.csv")
labeler1 = gold_standard["quote_use"]
labeler2 = gold_standard["quote_use"]
kappa = cohen_kappa_score(labeler1, labeler2)
print(kappa)