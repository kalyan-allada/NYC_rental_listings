# -*- coding: utf-8 -*-
# Predict a set of probabilities for every listing with the
# interest_level categories of "high", "medium" and "low"
# We use XGBoost classifier 

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# A generic function to run XGB model
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05  
    param['max_depth'] = 5
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 2   
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7 
    param['seed'] = seed_val
    #param['reg_alpha'] = 0
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# Read train and test json files
train_df = pd.read_json("data/train.json")
test_df = pd.read_json("data/test.json")
print("train set: "+ str(train_df.shape))
print("test set: "+ str(test_df.shape))

# --- Feature engineering ---
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# count of photos, # of features, # of description words
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])

# label encode (with numerical values) categorical features
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
      if train_df[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            #fit both train and test together to include all unique values
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            
# Build "manager_skill" feature and add to the train/test set
# Compute fractions and count for each manager
y_train_tmp = train_df["interest_level"]
tmp = pd.concat([train_df.manager_id,pd.get_dummies(y_train_tmp)], axis = 1).groupby('manager_id').mean()
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
# Add manager_skill to features to use
features_to_use.append("manager_skill")
#print(features_to_use)

# Join words in "features" and do token count
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
#print(train_df["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

# Stack train/test matrices with tfidf sparse matrix 
# and save them in compressed row format 
X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr() 

# Encode target labels
target_num_map = {'high':0, 'medium':1, 'low':2}
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

# Split train into train/valid sets
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.30, random_state=42)
#print(train_X.shape, valid_X.shape)
#print(train_y.shape, valid_y.shape)

# run XGB to predict and then validate
preds, model = runXGB(train_X, train_y, valid_X, valid_y)
print(log_loss(valid_y, preds))

# Plot feature importance
#xgb.plot_importance(model)

# Fit XGBoost model to train set and predict probabilities for test set
# Choose num_rounds based on the best iteration during validation above 
preds, model = runXGB(X, y, test_X, num_rounds=617)
out_df = pd.DataFrame(preds)
print(out_df.shape)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("predictions_xgb.csv", index=False)


            
            
