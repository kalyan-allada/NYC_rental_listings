# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import log_loss, classification_report
from sklearn import preprocessing
import sklearn.pipeline as pl

# Read json files
train_df = pd.read_json(open("data/train.json", "r"))
print(train_df.shape)
test_df = pd.read_json(open("data/test.json", "r"))
print(test_df.shape)

# Feature engineering
# Add features to train set
train_df["num_photos"] = train_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day

# Add features to test set
test_df["num_photos"] = test_df["photos"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
test_df["created"] = pd.to_datetime(test_df["created"])
test_df["created_year"] = test_df["created"].dt.year
test_df["created_month"] = test_df["created"].dt.month
test_df["created_day"] = test_df["created"].dt.day

# These are the features we will start with       
features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
 
# Encode label for categorical features
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for k in categorical:
      if train_df[k].dtype=='object':
            lab_enc = preprocessing.LabelEncoder()
            #fit both train and test sets together to include all unique values
            lab_enc.fit(list(train_df[k].values) + list(test_df[k].values))
            train_df[k] = lab_enc.transform(list(train_df[k].values))
            test_df[k] = lab_enc.transform(list(test_df[k].values))
            features_to_use.append(k)
      
X = train_df[features_to_use]
y = train_df["interest_level"]
print ("Following features will be used: " + str(features_to_use))

# Split the train set to train and validation sets for checks
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# We will use RFC (best parameters can be found from grid search below)
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_features='sqrt', min_samples_split=5)

# we will use pipelines to do the fitting and prediction..
# ..we can add more steps if necessary
steps = [('random_forest', clf)]
pipe_rfc = pl.Pipeline(steps)

# parameter grid search
#params = dict(random_forest__n_estimators=[1000, 1500, 2000],
#              random_forest__min_samples_split=[5, 6, 7],
#              random_forest__max_features=['sqrt', 'log2'])
#cv_rfc = GridSearchCV(pipe_rfc, param_grid=params, n_jobs=1, cv=3)
#cv_rfc.fit(X_train, y_train)
#print cv_rfc.best_params_


# fit train set and make prediction on validation set
pipe_rfc.fit(X_train, y_train)
y_val_pred = pipe_rfc.predict_proba(X_val)
print("Log loss for validation set: " + str(log_loss(y_val, y_val_pred)))

#Check cross validation accuracy
scores = cross_val_score(pipe_rfc, X, y, cv=3, n_jobs=1);
print ("Accuracy of all classes: " + str(np.mean(scores)))

#Now predict with test data
X_test = test_df[features_to_use]
#fit entire train data before predicting on test set
pipe_rfc.fit(X, y)
y_test = pipe_rfc.predict_proba(X_test)

# Let's visualize features importances
pd.Series(index = features_to_use, data = clf.feature_importances_).sort_values().plot(kind = 'bar')

# Write output to a csv file for submission
pred_df = pd.DataFrame()
label_id = {label: i for i, label in enumerate(clf.classes_)}
pred_df["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    pred_df[label] = y_test[:, label_id[label]]
pred_df.to_csv("predictions_rfc.csv", index=False)

