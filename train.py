#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import requests
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pickle

#get_ipython().run_line_magic('matplotlib', 'inline')

# Parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# Data preparation

file_path = 'Patients-Data-Heart-Disease-Prediction.xlsx'  # BE PATIENT, IT'S A LARGE FILE, IT TAKES MUCH TIME LOADING
df = pd.read_excel(file_path)

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.hadheartattack.values
y_val = df_val.hadheartattack.values
y_test = df_test.hadheartattack.values

del df_train['hadheartattack']
del df_val['hadheartattack']
del df_test['hadheartattack']


categorical = list(df.dtypes[df_full_train.dtypes == 'object'].index)
numerical = df_full_train.dtypes[df_full_train.dtypes != "object"]
numerical = list(numerical.index)
numerical.remove('patientid')
numerical.remove('hadheartattack')


# Training

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=237630) 
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Validation

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.hadheartattack.values
    y_val = df_val.hadheartattack.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1
    
print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Training the final model

print('training the final model')

dv, model = train(df_full_train, df_full_train.hadheartattack.values, C=1.0)
y_pred = predict(df_test, dv, model)
#y_test = df_test.hadheartattack.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# ### Save The Model

with open(output_file, 'wb') as f_out :
    pickle.dump((dv, model), f_out)
    

print(f'the model is saved to {output_file}')








