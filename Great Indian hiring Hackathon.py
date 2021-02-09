
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 
os.chdir(r"C:\Users\user\Desktop\projects and code\Hackathons\great indian hiring hack")

train  = pd.read_csv("Train.csv")
test   = pd.read_csv("Test.csv")
sample_submission=pd.read_csv("Sample Submission.csv")

train= train.drop(['InvoiceNo','CustomerID','Description'],axis = 1)#drop year 
test= test.drop(['InvoiceNo','CustomerID','Description'],axis = 1)

train.boxplot(column =['UnitPrice'], grid = False) 
train.boxplot(column =['Quantity'], grid = False) 

#removing outliers
train = train[train['UnitPrice'] <= 5.2]
train = train[train['UnitPrice'] > 0.25]

train = train[train['Quantity'] > -12]
train = train[train['Quantity'] <  27]

def f(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'

train['InvoiceDate'] = train['InvoiceDate'].astype('datetime64[ns]')
train['year']=train.InvoiceDate.dt.year
train['month']=train.InvoiceDate.dt.month
train['day']=train.InvoiceDate.dt.day
train['hour']=train.InvoiceDate.dt.hour
train['week_name']=train.InvoiceDate.dt.weekday_name
train['hour'] = train['hour'].apply(f)

train= train.drop('InvoiceDate',axis = 1)

train.boxplot(by ='month', column =['UnitPrice'], grid = False) 
train.boxplot(by ='month', column =['Quantity'], grid = False) 

#converting string to numeric
train['month'] = pd.factorize(train['month'])[0]
train['week_name'] = pd.factorize(train['week_name'])[0]
train['hour'] = pd.factorize(train['hour'])[0]
train['Country'] = pd.factorize(train['Country'])[0]
train['StockCode'] = pd.factorize(train['StockCode'])[0]

test['InvoiceDate'] = test['InvoiceDate'].astype('datetime64[ns]')
test['year']=test.InvoiceDate.dt.year
test['month']=test.InvoiceDate.dt.month
test['day']=test.InvoiceDate.dt.day
test['hour']=test.InvoiceDate.dt.hour
test['week_name']=test.InvoiceDate.dt.weekday_name
test['hour'] = test['hour'].apply(f)

test= test.drop('InvoiceDate',axis = 1)

#converting string to numeric
test['month'] = pd.factorize(test['month'])[0]
test['week_name'] = pd.factorize(test['week_name'])[0]
test['hour'] = pd.factorize(test['hour'])[0]
test['Country'] = pd.factorize(test['Country'])[0]
test['StockCode'] = pd.factorize(test['StockCode'])[0]
#Take targate variable into y

from sklearn.preprocessing import scale
data_standardized=scale(train)
data_standardized.mean(axis=0)

from sklearn.preprocessing import scale
data_standardized=scale(test)
data_standardized.mean(axis=0)

y = train['UnitPrice']
X = train.drop('UnitPrice',axis = 1)

from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(X)
normalizedX=scaler.transform(X)

# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X,y)

import xgboost
xgb=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
colsample_bytree=1, max_depth=7)

xgb.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression
lr_model=model = LinearRegression()

lr_model.fit(X, y)

from sklearn.neighbors import KNeighborsRegressor
# instantiate the model and set the number of neighbors to consider to 3
knn = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
knn.fit(X,y)

# Predicting results for test dataset
y_pred = rf.predict(test)
submission = pd.DataFrame({"UnitPrice": y_pred})
submission.to_csv('submission_rf_bbb.csv', index=False)

# Predicting results for test dataset
y_pred = lr_model.predict(test)
submission = pd.DataFrame({"UnitPrice": y_pred})
submission.to_csv('submission_lr_aaaa.csv', index=False)

# Predicting results for test dataset
y_pred = xgb.predict(test)
submission = pd.DataFrame({"UnitPrice": y_pred})
submission.to_csv('submission_xgb_aaaa.csv', index=False)

# Predicting results for test dataset
y_pred = knn.predict(test)
submission = pd.DataFrame({"UnitPrice": y_pred})
submission.to_csv('submission_knn.csv', index=False)