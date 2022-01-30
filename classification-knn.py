# Classification KNN - K Nearest Neighbors

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("teleCust1000t.csv")
print(df.head())

# data file has 4 different classes (custcat field). Find count for each
print(df['custcat'].value_counts())

#You can easily explore your data using visualization techniques:
#df.hist(column='income', bins=50)
#plt.show()

print(df.columns)

#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
print(X[0:5])

y = df['custcat'].values
print(y[0:5])

#Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

# Train Model and Predict
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
print(neigh)

# Predicting
yhat = neigh.predict(X_test)
print(yhat[0:5])

# Accuracy evaluation
from sklearn import metrics
print('Train set Accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, neigh.predict(X_test)))

#
# Practice w/ k=6
#
knn6 = KNeighborsClassifier(n_neighbors=6).fit(X_train,y_train)

# Accuracy evaluation
from sklearn import metrics
print('Train set Accuracy: ', metrics.accuracy_score(y_train, knn6.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, knn6.predict(X_test)))

# Loop over different K values
import numpy as np
Ks = 40
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)