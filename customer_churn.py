import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv('customer_churn_data_csv.csv')

#remove some irrelevant columns
df.drop(columns = ['CustomerID','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','Internet Service','Contract','Payment Method','Churn Label','Churn Score'], inplace = True)
#dataset is large so drop rows with empty spaces
df.dropna(inplace = True)
#drop duplicate rows
df.drop_duplicates(inplace = True)
#change yes or no to boolean values
df.replace({'Yes':int(1), 'No':int(0), 'yes':int(1), 'no':int(0), 'Female':int(0), 'Male':int(1), 'female':int(0), 'male':int(1)},inplace = True)

#create training sample
traindata = df.copy()
traindata.drop(traindata.index[::4],inplace = True)
traindata = traindata.astype(float)
#create test sample
testdata = df.copy()
testdata = testdata.iloc[::4]
testdata = testdata.astype(float)

#carry out model training and model testing
model = XGBClassifier(n_estimators = 100, max_depth = 5, learning_rate = 0.1, random_state = 42, eval_metric = 'logloss')
model.fit(traindata.drop(columns = 'Churn Value'),traindata.loc[:,'Churn Value'])
predictions = model.predict(testdata.drop(columns = 'Churn Value'))
print(classification_report(testdata.loc[:,'Churn Value'],predictions))