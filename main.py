#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:28:23 2021

@author: bartventer
"""

######################################################################
######### IMPORTING THE LIBRARIES ####################################
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

######################################################################
######### IMPORTING THE DATASET ######################################
######################################################################

df = pd.read_csv("lending_club_loan_two.csv")


######################################################################
######### EXPLORATORY DATA ANALYSES ##################################
######################################################################

#Explore the features and data types
df.info()

#Count Plot of dependant variable to see 
#whether the problem is balanced
sns.countplot(df['loan_status'])

#Explore correlation between the continues feature variables
df.corr()
#Visualise the feature correlation with a heatmap
plt.figure(figsize=(12,7))
sns.heatmap(data = df.corr(), annot=True, cmap='viridis')
plt.ylim(10, 0)

#Explore relationship between loan status and the loan amount
sns.boxplot(x='loan_status', y= 'loan_amnt', data=df)
#Explore summary statistics of the loan amount, grouped by loan_status
#This will assist in analysing the boxplot
df.groupby("loan_status")["loan_amnt"].describe()

#Explore the grade and subgrade features
#Grade:
sns.countplot(x='grade',data=df,hue='loan_status')
#Sub Grade per unique category:
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )
#Sub Grade per unique category differentiated with the loan status:
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')
#Looks like F and G subgrades don't get paid back that often. 
#Countplot just for those subgrades:
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')

#Creating a new column for the dependant variable as Loan repaid 
#(1 for repaid, 0 for charged off)
sns.countplot(df["loan_status"])
df['loan_status'].unique()   
df["loan_repaid"] = df["loan_status"].apply(lambda a: 1 if a == "Fully Paid" else 0)
df[["loan_status", "loan_repaid"]]

#Bar plot showing the correlation of the numeric features 
#to the new loan_repaid column. 
df.corr()["loan_repaid"].drop("loan_repaid").sort_values().plot(kind="bar")


######################################################################
######### DATA PREPROCESSING #########################################
######### MISSING DATA ###############################################
######################################################################

#Length of the dataframe
len(df)
#Total amount of missing values per column
df.isnull().sum()
#Missing values as percentage of column observations
df.isnull().sum()/len(df) *100

#Examining emp_title and emp_length to see whether it will be okay to drop them.
#Emp_title column:
df['emp_title'].nunique()
df['emp_title'].value_counts()
#Realistically there are too many unique job titles
# to try to convert this to a dummy variable feature. 
# Will remove the emp_title column
df = df.drop('emp_title',axis=1)

#Emp_length column:
#Count plot of the emp_length feature column. 
sorted(df['emp_length'].dropna().unique())

emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)
# Countplot with a hue separating Fully Paid vs Charged Off
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

# Percentage of charge offs per category. 
#Essentially illustrating what percent of people per employment category 
# didn't pay back their loan.
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp

emp_len
emp_len.plot(kind='bar')

#Charge off rates are extremely similar across all employment lengths
#Will drop the emp_length column
df = df.drop('emp_length',axis=1)

#Remaining features with missing data
df.isnull().sum()
df.isnull().sum()/len(df) *100

# Reviewing the title column vs the purpose column. To see if there's repeated information
df['purpose'].head(10)
df['title'].head(10)
# The title column is simply a string subcategory/description of the purpose column. 
# Will drop the title column
df = df.drop('title',axis=1)


#Create a value_counts of the mort_acc column
df['mort_acc'].value_counts()
# Correlation with the mort_acc column
df.corr()['mort_acc'].sort_values()

#Looks like the total_acc feature correlates with the mort_acc ,
#Will group the dataframe by the total_acc and calculate the mean value 
# for the mort_acc per total_acc entry.
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

#Imputing missing mortage account values
# Will fill in the missing mort_acc values based on their total_acc value. 
# If the mort_acc is missing, then will fill in that missing value with 
# the mean value corresponding to its total_acc value from the Series created above. 

def fill_mort_acc(total_acc,mort_acc):
    """
    Accept the total_acc and mort_acc values for the row.
    
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    """
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


#Remaining missing values
df.isnull().sum()
#revol_util and the pub_rec_bankruptcies have missing data points, 
#but they account for less than 0.5% of the total data. 
#Will remove the rows that are missing those values in those columns with dropna().
df = df.dropna()
df.isnull().sum()


######################################################################
######### DATA PREPROCESSING #########################################
######### ENCODING CATEGORICAL X FEATURES ############################
######################################################################

# Listing all the columns that are currently non-numeric.
df.info()
df.select_dtypes(['object']).columns

#Term Feature
#Will Convert the term feature into either a 36 or 60 integer numeric data type using .apply() 
df['term'].value_counts()
df['term'] = df['term'].apply(lambda t: int(t.split()[0]) )

#Grade feature
# Already know grade is part of sub_grade, so will drop the grade feature
df = df.drop('grade',axis=1)

#Subgrade
#Will Convert the subgrade into dummy variables. 
#Then concatenate the new columns to the original dataframe. 
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns

#Remaining object data types
df.select_dtypes(['object']).columns

# verification_status, application_type,initial_list_status,purpose 
# Will convert the columns: ['verification_status', 'application_type','initial_list_status','purpose'] 
#into dummy variables and concatenate them with the original dataframe. 
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

#home_ownership
# Value_counts for the home_ownership column
df['home_ownership'].value_counts()
# Will convert these to dummy variables, but 
# also replace NONE and ANY with OTHER, so that just 4 categories, MORTGAGE, RENT, OWN, OTHER. 
# Will then concatenate them with the original dataframe. 
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

# ### address
# Feature engineering; creating a column called 'zip_code' 
# that extracts the zip code from the address column
df["address"].head()
#Applying regex to extract zip code from address column
import re
pattern = re.compile(r"(\w{2}) (\d{5})\Z")
df["zip_code"] = df["address"].apply(lambda a: re.search(pattern, a).group(2) ) 
df["zip_code"].value_counts()

#Will turn zip_code column into dummy variables using pandas. 
#Will then Concatenate the result and drop the original zip_code column 
# along with dropping the address column

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


#issue_d 
#This would be data leakage, wouldn't know beforehand 
# whether or not a loan would be issued when using the model, 
# so in theory wouldn't have an issue_date.
#Will drop this feature.
df = df.drop('issue_d',axis=1)


#earliest_cr_line
#This appears to be a historical time stamp feature. 
# Will convert string dates to datetime objects
# Will then Extract the year from this feature using a .apply function, 
# Will then convert it to a numeric feature. 
# Will then set this new data to a feature column called 'earliest_cr_year'.
# Will then drop the earliest_cr_line feature.
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda d: d.year) 
df = df.drop('earliest_cr_line',axis=1)


#Remaining object columns
df.select_dtypes(['object']).columns
#Will drop loan_status column, as we already encoded the depedant variable
# in the loan_repaid column
df = df.drop('loan_status',axis=1)

#SAVE THE DATAFRAME
df_filename = "loan_repaid_df.h5"
df.to_hdf(df_filename, key = "df", mode = "w")
df = pd.read_hdf(df_filename, key = "df")


######################################################################
######### DATA PREPROCESSING #########################################
######### SPLITTING DATA INTO A TRAINING AND TEST SET ################
######################################################################

#SPLITTING DATA INTO THE INDEPENDANT X VARIABLES, AND DEPENDANT Y VARIABLES
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

#SPLITTING THE DATA INTO A TRAINING AND TEST SET (RANDOME_STATE IS PURELY SELECTED FOR CONSISTENCY IN SIMULATIONS)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


#APPLY FEATURE SCALING TO TRAINING SET
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

#SAVE THE SCALER
import joblib
scaler_filename = "loan_repaid_scaler.save"
joblib.dump(scaler, scaler_filename)
#LOAD THE SCALER
scaler = joblib.load(scaler_filename) 

#SCALE THE TEST SET
X_test = scaler.transform(X_test)


######################################################################
######### BUILDING AND TRAINING THE MODEL ############################
######################################################################

#CREATING THE CLASSIFIER ANN MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#CREATING THE ANN INFRASTRUCTURE
#INITIALISING THE MODEL
model = Sequential()

#ADDING THE INPUT LAYER WITH A DROPUT LAYER
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))

#HIDDEN LAYER
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))

#HIDDEN LAYER
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))

#OUTPUT LAYER
model.add(Dense(1, activation='sigmoid'))

#COMPILE THE MODEL
model.compile(optimizer="adam", loss="binary_crossentropy")

#TRAINING THE MODEL
#INITIALISE THE EARLY STOP CALLBACK OBJECT
early_stop = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=10)

#MODEL TRAINING
model.fit(x=X_train,
          y=y_train,
          epochs=150,
          callbacks=[early_stop],
          validation_data=(X_test, y_test),
          batch_size=256)

#SAVE TRAINING HISTORY
losses = pd.DataFrame(model.history.history)
losses_filename = "loan_repaid_history.csv" 
losses.to_csv(losses_filename, index=False)
losses = pd.read_csv(losses_filename)

#SAVE THE MODEL
from tensorflow.keras.models import load_model
model_filename = "loan_repaid_ann_classifier_model.h5"
model.save(model_filename)
model = load_model(model_filename)

#VISAULISE THE TRAINING AND VALIDATION LOSSES
losses.plot()


######################################################################
############ TEST PREDICTIONS AND MODEL EVALUATION ###################
######################################################################

#Model predictions
predictions = model.predict_classes(X_test)

#Evaluating the predictions
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

######################################################################
############ MAKE A NEW PREDICTION ###################################
######################################################################

#PREDICT A RANDOM SAMPLE FROM THE DATASET
import random
random_ind = random.randint(0,len(df))
new_customer = df.iloc[random_ind]
new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind].values.reshape(1,-1)
new_customer = scaler.transform(new_customer)
model.predict_classes(new_customer)
df.iloc[random_ind]['loan_repaid']

