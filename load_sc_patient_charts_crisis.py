import pandas as pd

ICDs = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip',
    header=0, sep=',', quotechar='"')
peek_ICDs = ICDs.head()
print(peek_ICDs)
ICDs.info()
#Filter for ICD codes for 282.60-282.69, referring to sickle cell types w/wo crisis
#2824 for thalassemia w + w/o crisis (282.41-282.42)
is_SCD = ICDs[ICDs['ICD9_CODE'].str.contains('28262|28264|28269|28242', na=False)]

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
# Find unique chart types and decide to limit them
    # Remove Radiology, ECG, Respiratory, Echo notes
SC_NOTES_PATIENTS_FILTERED = SC_NOTES_PATIENTS[SC_NOTES_PATIENTS["CATEGORY"].str.contains("Radiology|ECG|Respiratory|Echo")==False]
SC_NOTES_PATIENTS_FILTERED.to_csv("SC_NOTES_PATIENTS_CRISIS.csv")


patients_unique_with_notes = SC_NOTES_PATIENTS_FILTERED.nunique()
# Can we compute how long the text is word-wise?
# Re run with a column that preserves ICD-09 code in final dataset to double check
# compare against CF patients?
# Way to be able to abstract race?