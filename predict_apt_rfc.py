# -*- coding: utf-8 -*-
# Predict a set of probabilities for every listing with the
# interest_level categories of "high", "medium" and "low"
# We use Random forest classifier 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.metrics import log_loss, classification_report
from sklearn import preprocessing
import sklearn.pipeline as pl
from sklearn import linear_model

# Read json files
train_df = pd.read_json(open("data/train.json", "r"))
print(train_df.shape)
test_df = pd.read_json(open("data/test.json", "r"))
print(test_df.shape)

# Feature engineering
# Add no. of photos, features,description words and created year/month/day/hour 
train_df["num_photos"] = train_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour

# Add these features to test set
test_df["num_photos"] = test_df["photos"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
test_df["created"] = pd.to_datetime(test_df["created"])
test_df["created_year"] = test_df["created"].dt.year
test_df["created_month"] = test_df["created"].dt.month
test_df["created_day"] = test_df["created"].dt.day
test_df["created_hour"] = test_df["created"].dt.hour

# These are the features to start
features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day", "created_hour", "listing_id"]
 
# Lable encode categorical features
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for k in categorical:
      if train_df[k].dtype=='object':
            lab_enc = preprocessing.LabelEncoder()
            #fit both train and test sets together to include all unique values
            lab_enc.fit(list(train_df[k].values) + list(test_df[k].values))
            train_df[k] = lab_enc.transform(list(train_df[k].values))
            test_df[k] = lab_enc.transform(list(test_df[k].values))
            features_to_use.append(k)

# Build "manager_skill" feature and add to the train/test set
# Compute fractions and count for each manager
y_train_tmp = train_df["interest_level"]
tmp = pd.concat([train_df.manager_id, pd.get_dummies(y_train_tmp)], axis = 1).groupby('manager_id').mean()
tmp.columns = ['high_frac','low_frac', 'medium_frac']
tmp['count'] = train_df.groupby('manager_id').count().iloc[:,1]
# Compute manager skill
tmp['manager_skill'] = tmp['high_frac']*2 + tmp['medium_frac']    
# Get indices for unranked managers
unranked_mgr_idx = tmp['count']<20
# Get indices for ranked ones
ranked_mgr_idx = ~unranked_mgr_idx
# Compute mean values from ranked managers and assign them to unranked ones
mean_values = tmp.loc[ranked_mgr_idx, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
tmp.loc[unranked_mgr_idx,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
# Merge tmp dataframe with train/test  
train_df = train_df.merge(tmp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
test_df = test_df.merge(tmp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
# Fill the null values for new managers in the test set with mean values
new_mgr_idx = test_df['high_frac'].isnull()
test_df.loc[new_mgr_idx,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
features_to_use.append("manager_skill")

print ("Following features will be used: " + str(features_to_use))

X = train_df[features_to_use]
X_test = test_df[features_to_use]
y = train_df["interest_level"]

# Split the train set to train and validation sets for checks
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=24)

# Initialize random forest classifier 
clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, max_features='sqrt', min_samples_split=5)

#Use pipeline steps - can add more steps if necessary
steps = [('random_forest', clf)]
pipe_rfc = pl.Pipeline(steps)

# Parameter grid search
#params = dict(random_forest__n_estimators=[200, 1000, 2000],
#              random_forest__min_samples_split=[5, 6, 7],
#              random_forest__max_features=['sqrt', 'log2'])
#cv_rfc = GridSearchCV(pipe_rfc, param_grid=params, n_jobs=1, cv=3)
#cv_rfc.fit(X_train, y_train)
#print cv_rfc.best_params_

# Fit train set and make prediction on validation set
pipe_rfc.fit(X_train, y_train)
y_val_pred = pipe_rfc.predict_proba(X_val)
print("Log loss for validation set: " + str(log_loss(y_val, y_val_pred)))

# Cross-validation and check scores (alternative to hold-out test above)
#cv_ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
#scores = cross_val_score(pipe_rfc, X, y, cv=cv_ss);
#print ("Accuracy of all classes: " + str(np.mean(scores)))

# Now fit full train set and make prediction on test set
pipe_rfc.fit(X, y)
pred_prob = pipe_rfc.predict_proba(X_test)

# Visualize features importances and make histogram of probabilities 
plt.figure(figsize=(12,8))
pd.Series(index = features_to_use, data = clf.feature_importances_).sort_values().plot(kind = 'bar')
plt.figure(figsize=(12,8))
plt.hist(pred_prob[:,0], bins= 200, label='Interest level = High')
plt.hist(pred_prob[:,1], bins= 200, label='Interest level = Low')
plt.hist(pred_prob[:,2], bins= 200, label='Interest level = Medium')
plt.legend()
plt.ylabel('Number of instances')
plt.xlabel('Probability for interest level')
plt.savefig('predicted_probs.png') 

# Write output to a file for submission
pred_df = pd.DataFrame()
label_id = {label: i for i, label in enumerate(clf.classes_)}
pred_df["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    pred_df[label] = pred_prob[:, label_id[label]]
pred_df.to_csv("predictions_rfc.csv", index=False)

#public LB: 0.58066