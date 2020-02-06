# BOSTON HOUSE PRICES - USING BACKWARD ELIMINATION

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
boston=pd.read_csv('Boston_House_Prices.csv')
X=boston.iloc[:,:-1].values
Y=boston.iloc[:,13].values

#spitting the dataset into train set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#fitting the model into train set
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)

#predicting the results of test set
Y_pred=model.predict(X_test)

#building optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)

#starting backward elimination
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
model_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
model_OLS.summary()
X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]
model_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
model_OLS.summary()
X_opt=X[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
model_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
model_OLS.summary()
#model is built