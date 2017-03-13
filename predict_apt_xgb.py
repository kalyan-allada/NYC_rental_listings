# -*- coding: utf-8 -*-
#This script is a modified version of a starter code provided by kaggle user SRK
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# define xgboost function
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

    plist = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plist, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plist, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


train_df = pd.read_json("data/train.json")
test_df = pd.read_json("data/test.json")

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# Feature engineering

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
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

# Convert list of strings to single string and apply count vectorizer on top            
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

# Combine the "features" matrix with rest of the features matrix
# and convert to a row type
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

# Get the target variable
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

# comment this part once validation is done and num_round has been optimized
#cv_scores = []
#kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1000)
#for dev_index, val_index in kf.split(range(train_X.shape[0])):
#        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
#        dev_y, val_y = train_y[dev_index], train_y[val_index]
#        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
#        cv_scores.append(log_loss(val_y, preds))
#        print(cv_scores)
#        break

# Choose num_rounds greater than best number from kfold validation above    
# since now we are using the total train set
preds, model = runXGB(train_X, train_y, test_X, num_rounds=1500)
out_df = pd.DataFrame(preds)
print(out_df.shape) 
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("predictions_xgb.csv", index=False)



            
            
