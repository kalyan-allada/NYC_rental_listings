## Prediction of customer's interest level for rental listings in NYC area

The goal of this project is to predict the number of inquiries a new rental listing receives based on the listing’s creation date and other features. The numebr of inquiries for a particular listings are given to us in the form of "interest_level" feature. The project is based on a kaggle competition. The description of the competition can be found [here](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries). We will build a machine learning mode to predict the interest level for every listing with three catagories, i.e, "low", "medium" and high. This is a multi-class classification problem where we are suppose to predict probability for each class for all the listings.

#### Files:
- **exploratory_analysis.ipynb** : A python nobebook for exploring the data and building features as necessary
- **predict_apt_rfc.py** : Script to perform prediction using the random forest classifier
- **prediction_apt_xgb** : Script to perform prediction using XGBoost module
- **predictions_rfc.csv** : Output file with predictions (RFC)
- **predictions_xgb.csv** : Output file with predictions (XGB)
- **README.md** : You are reading this file
- **data** : This is the data folder, not included here. The data can be downloaded from this [link](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data). A description of the data fields are also present at this link.

