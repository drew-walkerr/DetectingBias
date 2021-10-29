import pandas as pd

PATIENTS = pd.read_csv('PATIENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_patients = PATIENTS.head()
print(peek_patients)
PATIENTS.info()
ADMISSIONS = pd.read_csv('ADMISSIONS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_admissions = ADMISSIONS.head()
print(peek_admissions)
ADMISSIONS.info()