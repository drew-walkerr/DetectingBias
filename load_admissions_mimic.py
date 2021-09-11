import pandas as pd

ICDs = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_ICDs = ICDs.head()
print(peek_ICDs)
ICDs.info()
is_SCD = ICDs[ICDs['ICD9_CODE'].str.contains('2826',na=False)]

print(is_SCD.head())

is_SCD.info()

patients_unique = is_SCD['SUBJECT_ID'].drop_duplicates()

NOTES = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_NOTES = NOTES.head()
print(peek_NOTES)
NOTES.info()

SC_NOTES = pd.merge(NOTES,patients_unique, on = 'SUBJECT_ID', how = 'right')

print(SC_NOTES['TEXT'])

print(SC_NOTES.head())

