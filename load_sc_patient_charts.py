import pandas as pd

ICDs = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_ICDs = ICDs.head()
print(peek_ICDs)
ICDs.info()
is_SCD = ICDs[ICDs['ICD9_CODE'].str.contains('2826', na=False)]

print(is_SCD.head())

is_SCD.info()

patients_unique = is_SCD['SUBJECT_ID'].drop_duplicates()

NOTES = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_NOTES = NOTES.head()
print(peek_NOTES)
NOTES.info()
SC_NOTES = NOTES.merge(patients_unique, on = 'SUBJECT_ID')
print(SC_NOTES['TEXT'])
print(SC_NOTES.head())

SC_NOTES.info()

PATIENTS = pd.read_csv('PATIENTS.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_patients = PATIENTS.head()
print(peek_patients)

SC_NOTES_PATIENTS = SC_NOTES.merge(PATIENTS, on = 'SUBJECT_ID')

SC_NOTES_PATIENTS.info()

SC_NOTES_PATIENTS.to_csv("SC_NOTES_PATIENTS.csv")

# Find unique count of patients and patient-relevant covariates
# Find unique chart types and decide to limit them
    # Remove Radiology, ECG, Respiratory
# Can we compute how long the text is word-wise?
# Filter down to