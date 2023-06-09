import pandas as pd
from ydata_profiling import ProfileReport
df = pd.read_excel("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/ahs_insurance_sample.xlsx")
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from pandas_profiling import ProfileReport
from sklearn.base import clone
import os
from tpot import TPOTClassifier
from tpot import TPOTRegressor

#df.to_csv("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/insurance_sample.csv")

df['ID'] = range(1, len(df)+1)


df['BUYI'].corr(df['IFFEE'])
df['IFFEE'].value_counts()
df['BUYI'].value_counts()

#ProfileReport(df=df, title="Insurnace").to_file("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/AfterPrepro_Insurance-profiling.html")


# Preprocessing

# AMTI (insurance amount) negative values are worrysome, replace them with zeros(not applicable)
# Perhaps it is wrong to replace them with the median values since it could be some of the
# non respondents actually did not buy insurance.

df.loc[df['AMTI'] <=0, 'AMTI'] = 0

df.loc[df['CONFEE'] <=0, 'CONFEE'] = 0

df['CONFEE'].value_counts()
len(df['CONFEE'])


df['IFFEE'] = df['IFFEE'].replace(2, 0)

# Replace all occurrences of 93 with the median

HHAGE_median = df['HHAGE'].median()
df['HHAGE'] = df['HHAGE'].replace(93, HHAGE_median)
df.loc[df['HHAGE'] < 18, 'HHAGE'] = HHAGE_median

df['ZINC2'].value_counts()

sum((df['ZINC'].isin([-6, -7, -8, -9]))*1)

# Expected income in the next 12 months, this won't be available at later time of the prediction,
# Hence it should be excluded unless we get yearly data on expected income in each year.
del df['ZINCN']

# Household income is highly correlated with family income drop it.
sum(df['ZINC'])/ sum(df['ZINC2'])
del df['ZINC2']

#df['ZINCN_MISSING'] = df['ZINCN'].isin([-6, -7, -8, -9]).astype(int)

# REGION HOT ENCODING
region_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
df['REGION'] = df['REGION'].replace(region_map)
df_encoded = pd.get_dummies(df["REGION"])
df = df.join(df_encoded)
del df['REGION']




## METRO3 worrying it supposed to contain categirical values from 1 to 5 altough it only contains 
# 1 and 2. and9 very few 9.
# Replace 

# HOT ENCODING METRO
metro_map = {1: 'Central_city_of_MSA',
             2: 'City_urban_MSA',
             9: "Missing_M"}

df['METRO3'] = df['METRO3'].replace(metro_map)
df_encoded = pd.get_dummies(df["METRO3"])
del df_encoded['Missing_M']
df = df.join(df_encoded)
del df['METRO3']

df.loc[df['CONDO'] == 3, 'CONDO'] = 0

df['BUYI'].value_counts()

UNITSF_median = df['UNITSF'].median()
df.loc[df['UNITSF'] <= 0, 'UNITSF'] = UNITSF_median

df['LOT'].value_counts()

LOT_median = df['LOT'].median()
df.loc[df['LOT'] <= 0, 'LOT'] = 0
df.loc[df['LOT'] >= 20000, 'LOT'] = LOT_median

df['AVG_TOTAL_SQR'] = df[['LOT','UNITSF']].mean(axis=1)

"Dropp cellar might not be infromative"
del df['CELLAR']
"""
cellar_map ={1: "Full_basement",
2: "Partly_basement"
3: "crawl_space",
4: "concrete slab",
5: "other way"
}

df['CELLAR'] = df['CELLAR'].replace(cellar_map)
df_encoded = pd.get_dummies(df["CELLAR"])
df = df.join(df_encoded)
"""
# many missing values, does'nt make sense dropp it -> Referring to mobile homes?
del df['MOBILTYP']

inspect_df = df[df['TYPE'] == 1]

# Talk about House apartment 1, are our main interest, alternatively rows could be exclodued like cave, tent, those are irrational for housing, however since they are extemely few values,
# I did not exclude them.

df['BUYI'].value_counts()
df['TYPE']





del df['CLIMB']
# Does not matter the which floor
del df['FRSTOC']
# Does not matter if it is first 2nd or 3rd owned.
# Many missing values

df.loc[df['EVROD'] == 2, 'EVROD'] = 0
df.loc[df['EROACH'] == 2, 'EROACH'] = 0
df.loc[df['CRACKS'] == 2, 'CRACKS'] = 0
df.loc[df['HOLES'] == 2, 'HOLES'] = 0
df['WINTERNONE'].fillna(0, inplace=True)


df.loc[df['AIR'] == 2, 'AIR'] = 0
df.loc[df['AIRSYS'] == 2, 'AIRSYS'] = 0


df['BAD_SHAPE'] = df[['EVROD', 'EROACH', 'CRACKS','HOLES']].mean(axis=1)


df['Preventable_Loss'] = df['BUYI'] * df['AMTI'] * 0.3
df['Avoidable_labor_cost'] = abs(df['BUYI']-1) * 500

# If fee needs to be payed to the ooperative, homeowner's association,
# Nan values high correlation, might not be informative.
del df['IFFEE']

# We don't know if we will have the data one year from now, drop it 
del df['ZINCH']

#X.to_csv("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/X_trasnformed_insurance_sample.csv")

X = df.copy(deep=True)
y = X.pop("BUYI")


#  Leakage
Leakage = df[['AMTI', 'Preventable_Loss', 'Avoidable_labor_cost']].copy(deep=True)

# BUILT is a time variable, it is available at prediction, altough we should be careful and not use it for any calculation.
 

# dropping the leaky
X = X.drop(columns=['AMTI','Preventable_Loss','Avoidable_labor_cost'])


nans_per_column = X.isna().sum()
nans_per_column


y.value_counts()
# High imbalance split 50-50%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=77, test_size= 0.50, stratify=y) # 50% of the data as training

X_value, X_test, y_value, y_test = train_test_split(
    X_test, y_test, random_state=77, test_size= 0.50, stratify=y_test)  # 50% of the trainiig data(25% - 25% of the entire dataset)

# check distribution, it's the same, we are safe
default_distribution = y.value_counts() # distribution of 0-s and y-s in the dataframe.

y_test.value_counts() / sum(y_test.value_counts())
default_distribution / sum(default_distribution)


# Apply SMOTE to create synthetic observations.
# We resample the training dataset and balalnce the observations with SMOTE.
X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)
# equally distributed 50% - 50%
y_smote.value_counts()  / sum(y_smote.value_counts())


INS_DR = X_smote.copy(deep=True)
INS_DR['BUYI']= y_smote
INS_DR.to_excel("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/INS_SMOTE_50.xlsx", index=False, header=True)


INS_DR_value = X_value.copy(deep=True)
INS_DR_value['BUYI'] = y_value
#write this to Excel so we can upload it to DR and do some work with it
INS_DR_value.to_excel("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/INS_DR_value.xlsx", index=False, header=True)


pd.read_csv("/Users/student/Documents/UNI/2023 Spring/Machine Learning/Final EXAM/predictions/DR_VAL_predictions_INS_SMOTE_50.csv")

rus = RandomUnderSampler(sampling_strategy=0.85)
X_downsampled, y_downsampled = rus.fit_resample(X_train, y_train)


len(X_downsampled)
len(y_downsampled)
X_downsampled.value_counts() / sum(y_downsampled.value_counts())


y_downsampled.value_counts()


ros = RandomOverSampler(random_state=42, sampling_strategy=.10)

# Fit the random over-sampling object to the data
X_upsampled, y_upsampled = ros.fit_resample(X_train, y_train)

y_upsampled.value_counts()

clf = TPOTClassifier(generations=5, population_size=100, verbosity=2, n_jobs=-1)
clf_upsampled = clone(clf, safe=True)

clf_upsampled.fit(X_downsampled,y_downsampled)

model = clf_upsampled.fitted_pipeline_

clf_SMOTE = clone(clf, safe=True)
clf_SMOTE.fit(X_smote, y_smote) 

confusion_matrix(y_train, clf_SMOTE.predict(X_train))
  

