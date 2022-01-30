import pandas as pd

drug = pd.read_csv("drug200.csv")
print(drug.head())

X = drug[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y = drug['Drug']
print(X[0:5])
print(Y[0:5])
#print(X)
#print(Y)

"""
some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn Decision Trees do not handle categorical variables.
But still we can convert these features to numerical values. pandas.get_dummies() Convert categorical variable into dummy/indicator variables.
"""
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

"""
Setting up the Decision Tree

train_test_split will return 4 different parameters.
The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.
The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.

"""
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

"""
   Modeling
"""
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth= 4)
print(drugTree)

#fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset, y_trainset)

"""
   Prediction
"""
predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])


"""
   Evaluation
"""
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


"""
   Visualization
"""
