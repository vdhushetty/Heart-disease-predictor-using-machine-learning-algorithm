import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from mlxtend.plotting import plot_decision_regions
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron            
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

heart_data = pd.read_csv("C:/Users/bhaga/Desktop/EEE 591 Python/Project 1/heart1.csv")
heart_data

X = heart_data.iloc[:, 0:13]
y = heart_data.iloc[:, 13]
X_train, X_test, y_train, y_test =          train_test_split(X,y,test_size=0.3,random_state=0)


sc = StandardScaler()                    # create the standard scalar
sc.fit(X_train)                          # compute the required transformation
X_train_std = sc.transform(X_train)      # apply to the training data
X_test_std = sc.transform(X_test)

#PERCEPTRON

ppn = Perceptron(max_iter=7, tol=1e-3, eta0=0.004,fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train)           
print('Number in test ',len(y_test))
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# we did the stack so we can see how the combination of test and train data did
y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples: %d' %        (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


# Logistic Regression


lr = LogisticRegression(C=9, solver='liblinear',                         multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)         # apply the algorithm to training data
y_pred_lr = lr.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred_lr).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_lr))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_lr = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
y_combined_pred_lr = lr.predict(X_combined_std)
print('Misclassified combined samples: %d' %       (y_combined != y_combined_pred_lr).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_lr, y_combined_pred_lr))


# Support Vector Machines


for c_val in [0.1,1.0,10.0]:
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(X_train_std, y_train)                      
    y_pred = svm.predict(X_test_std)                   # work on the test data
    # show the results
    print("Results for C =",c_val)
    print('Number in test ',len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    # combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    # and analyze the combined sets
    print('Number in combined ',len(y_combined))
    y_combined_pred = svm.predict(X_combined_std)
    print('Misclassified combined samples: %d' %           (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' %            accuracy_score(y_combined, y_combined_pred))



# DECISION TREE LEARNING

tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)  # defining decision tree learning
tree.fit(X_train, y_train)  # fitting the data
y_pred_dt = tree.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred_dt).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_dt))
    # combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_dt = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_dt))
y_combined_pred_dt = tree.predict(X_combined_std)  # Predicting x-combined outcome
# Printing the accuracy score
print('Misclassified combined samples: %d' %       (y_combined_dt != y_combined_pred_dt).sum())
print('Combined Accuracy: %.2f' %       accuracy_score(y_combined_dt, y_combined_pred_dt))



# RANDOM FOREST

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=4)  # defining random forest
forest.fit(X_train, y_train)  # fitting the data
y_pred_rf = forest.predict(X_test)  # predicting X_test outcome
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred_rf).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_rf))
    # combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_rf = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_dt))
y_combined_pred_rf = forest.predict(X_combined_std)  # Predicting Y_combined outcome
# Printing the accuracy score
print('Misclassified combined samples: %d' %       (y_combined_rf != y_combined_pred_rf).sum())
print('Random Forest Accuracy: %.2f' %       accuracy_score(y_combined_rf, y_combined_pred_rf))



# K-NEAREST NEIGHBOUR

knn = KNeighborsClassifier(n_neighbors=40, p=2, metric='minkowski')  # defining KNN
knn.fit(X_train_std, y_train)  # fitting the data
y_pred_knn = knn.predict(X_test_std)  # predicting the X_test_std outcome
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred_knn).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_knn))
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_knn = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_knn))
y_combined_pred_knn = knn.predict(X_combined_std)  
print('KNN Accuracy: %.2f' % accuracy_score(y_combined_knn, y_combined_pred_knn))