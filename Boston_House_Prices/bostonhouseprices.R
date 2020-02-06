# BOSTON HOUSE PRICES - USING BACKWARD ELIMINATION

#import dataset
boston=read.csv('Boston_House_Prices.csv')

#splitting dataset into test and train sets
set.seed(2610)
split=sample.split(boston$PRICE, SplitRatio = 0.8)
train_set=subset(boston, split==TRUE)
test_set=subset(boston, split==FALSE)

#fitting multiple linear regression in training set
regressor = lm(formula = PRICE ~.,
               data=train_set)
summary(regressor)

#predicting the test set results
Y_pred = predict(regressor,newdata = test_set)

#backward elimination
regressor = lm(formula = PRICE ~ CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+RAD+TAX+PTRATIO+B+LSTAT,
               data=train_set)
summary(regressor)
regressor = lm(formula = PRICE ~ CRIM+ZN+INDUS+CHAS+NOX+RM+DIS+RAD+TAX+PTRATIO+B+LSTAT,
               data=train_set)
summary(regressor)
regressor = lm(formula = PRICE ~ CRIM+ZN+CHAS+NOX+RM+DIS+RAD+TAX+PTRATIO+B+LSTAT,
               data=train_set)
summary(regressor)
