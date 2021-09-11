import pandas as pd

PATIENTS = pd.read_csv('PATIENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_patients = PATIENTS.head()
print(peek_patients)

PATIENTS.info()