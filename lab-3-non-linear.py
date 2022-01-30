import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")
print(df.head().to_string())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
print(msk)
train = cdf[msk]
test = cdf[~msk]

# how we can fit our data on the polynomial equation while we have only parameter?
# PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set.

print("\n\n Square \n\n")
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, train_y)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Evaluation
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )



# Cubic
print("\n\n Cubic \n\n")
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly3 = PolynomialFeatures(degree=3)
train_x_poly = poly3.fit_transform(train_x)

from sklearn import linear_model
cubicModel = linear_model.LinearRegression()
cubicModel.fit(train_x_poly, train_y)

print('Coefficients: ', cubicModel.coef_)
print('Intercept: ', cubicModel.intercept_)

import matplotlib.pyplot as plt
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = cubicModel.intercept_[0] + cubicModel.coef_[0][1]*XX + cubicModel.coef_[0][2]*np.power(XX,2) + cubicModel.coef_[0][3]*np.power(XX,3)
plt.plot(XX,yy,'-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

from sklearn.metrics import r2_score
test_poly_3 = poly3.fit_transform(test_x)
test_y_ = cubicModel.predict(test_poly_3)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean(test_y_ - test_y) ** 2)
print("R2-score: %.2f" % r2_score(test_y_, test_y))

