import copy
import pickle
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

# First load all the datasets
x_train_l1 = pd.read_csv('ProcessedWithPCA/x_train_l1_pca.txt', header=0)
x_train_l2 = pd.read_csv('ProcessedWithPCA/x_train_l2_pca.txt', header=0)

x_test_l1 = pd.read_csv('ProcessedWithPCA/x_test_l1_pca.txt', header=0)
x_test_l2 = pd.read_csv('ProcessedWithPCA/x_test_l2_pca.txt', header=0)

# then load the target variables
y_train_l1 = np.load('ProcessedWithPCA/y_train_l1.npy')
y_train_l2 = np.load('ProcessedWithPCA/y_train_l2.npy')

y_test_l1 = np.load('ProcessedWithPCA/y_test_l1.npy')
y_test_l2 = np.load('ProcessedWithPCA/y_test_l2.npy')

# The full targets are loaded in y_test_full.npy
x_test_l1_full = pd.read_csv("ProcessedWithPCA/x_test_l1_pca.txt", header=0)
x_test_l2_full = pd.read_csv("ProcessedWithPCA/x_test_l2_pca.txt", header=0)

y_test_l2_full = np.load('AdditionalSets/l2_full_test_targets.npy')

# print shapes of datasets
print("Shape of x_train_l1:", x_train_l1.shape)
print("Shape of x_train_l2:", x_train_l2.shape)
print("Shape of x_test_l1:", x_test_l1.shape)
print("Shape of x_test_l2:", x_test_l2.shape)
print("Shape of y_train_l1:", y_train_l1.shape)
print("Shape of y_train_l2:", y_train_l2.shape)
print("Shape of y_test_l1:", y_test_l1.shape)
print("Shape of y_test_l2:", y_test_l2.shape)

# then we see if all dimensions match
assert x_train_l1.shape[0] == y_train_l1.shape[0], "Number of rows in x_train_l1 and y_train_l1 do not match"
assert x_train_l2.shape[0] == y_train_l2.shape[0], "Number of rows in x_train_l2 and y_train_l2 do not match"
assert x_test_l1.shape[0] == y_test_l1.shape[0], "Number of rows in x_test_l1 and y_test_l1 do not match"
assert x_test_l2.shape[0] == y_test_l2.shape[0], "Number of rows in x_test_l2 and y_test_l2 do not match"

# now we create the actual models with default parameters


print("LAYER 1\n")

print(x_train_l1.shape, y_train_l1.shape)

def nbc():
    print("\nNBC")
    l1_clf = GaussianNB()

    start = datetime.now()
    l1_clf.fit(x_train_l1, y_train_l1)
    end = datetime.now() - start

    predicted = l1_clf.predict(x_test_l1)

    print('Metrics for layer 1:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l1, predicted))
    print('Accuracy = ', accuracy_score(y_test_l1, predicted))
    print('F1 Score = ', f1_score(y_test_l1, predicted))
    print('Precision = ', precision_score(y_test_l1, predicted))
    print('Recall = ', recall_score(y_test_l1 ,predicted))
    print('Train time: ', end)

    return l1_clf

def rf():
    print("\nRandomForest")
    l1_clf = RandomForestClassifier()

    start = datetime.now()
    l1_clf.fit(x_train_l1, y_train_l1)
    end = datetime.now() - start

    predicted = l1_clf.predict(x_test_l1)

    print('Metrics for layer 1:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l1, predicted))
    print('Accuracy = ', accuracy_score(y_test_l1, predicted))
    print('F1 Score = ', f1_score(y_test_l1, predicted))
    print('Precision = ', precision_score(y_test_l1, predicted))
    print('Recall = ', recall_score(y_test_l1 ,predicted))
    print('Train time: ', end)

    return l1_clf

def svm():
    print("\nSVM")
    l1_svm = SVC()

    start = datetime.now()
    l1_clf.fit(x_train_l1, y_train_l1)
    end = datetime.now() - start

    predicted = l1_clf.predict(x_test_l1)

    print('Metrics for layer 1:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l1, predicted))
    print('Accuracy = ', accuracy_score(y_test_l1, predicted))
    print('F1 Score = ', f1_score(y_test_l1, predicted))
    print('Precision = ', precision_score(y_test_l1, predicted))
    print('Recall = ', recall_score(y_test_l1 ,predicted))
    print('Train time: ', end)

    return l1_clf

def mlp():
    print("\nMultiLayerPerceptron")
    l1_clf = MLPClassifier()

    start = datetime.now()
    l1_clf.fit(x_train_l1, y_train_l1)
    end = datetime.now() - start

    predicted = l1_clf.predict(x_test_l1)

    print('Metrics for layer 1:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l1, predicted))
    print('Accuracy = ', accuracy_score(y_test_l1, predicted))
    print('F1 Score = ', f1_score(y_test_l1, predicted))
    print('Precision = ', precision_score(y_test_l1, predicted))
    print('Recall = ', recall_score(y_test_l1 ,predicted))
    print('Train time: ', end)

    return l1_clf

def gbc():
    print("\nGradientBoosting")
    l1_clf = HistGradientBoostingClassifier()

    start = datetime.now()
    l1_clf.fit(x_train_l1, y_train_l1)
    end = datetime.now() - start

    predicted = l1_clf.predict(x_test_l1)

    print('Metrics for layer 1:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l1, predicted))
    print('Accuracy = ', accuracy_score(y_test_l1, predicted))
    print('F1 Score = ', f1_score(y_test_l1, predicted))
    print('Precision = ', precision_score(y_test_l1, predicted))
    print('Recall = ', recall_score(y_test_l1, predicted))
    print('Train time: ', end)

    return l1_clf

print("LAYER 2\n")

print(x_train_l2.shape, y_train_l2.shape)

def svm2():
    print("\nSVM")
    l2_clf = SVC(C=0.1, gamma=0.01, kernel='rbf', probability=True)

    start = datetime.now()
    l2_clf.fit(x_train_l2, y_train_l2)
    end = datetime.now() - start

    predicted = l2_clf.predict(x_test_l2)


    print('test metrics layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l2, predicted))
    print('Accuracy = ', accuracy_score(y_test_l2, predicted))
    print('F1 Score = ', f1_score(y_test_l2, predicted))
    print('Precision = ', precision_score(y_test_l2, predicted))
    print('Recall = ', recall_score(y_test_l2, predicted))
    print('Train time: ', end)

    predicted = l2_clf.predict(x_train_l2)

    print('\n\ntrain metrics layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_train_l2, predicted))
    print('Accuracy = ', accuracy_score(y_train_l2, predicted))
    print('F1 Score = ', f1_score(y_train_l2, predicted))
    print('Precision = ', precision_score(y_train_l2, predicted))
    print('Recall = ', recall_score(y_train_l2, predicted))

    return l2_clf

def nbc2():
    print("\nNBC")
    l2_clf = LogisticRegression()

    start = datetime.now()
    l2_clf.fit(x_train_l2, y_train_l2)
    end = datetime.now() - start

    predicted = l2_clf.predict(x_test_l2)

    print('Metrics for layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l2, predicted))
    print('Accuracy = ', accuracy_score(y_test_l2, predicted))
    print('F1 Score = ', f1_score(y_test_l2, predicted))
    print('Precision = ', precision_score(y_test_l2, predicted))
    print('Recall = ', recall_score(y_test_l2, predicted))
    print('Train time: ', end)

    predicted = l2_clf.predict(x_train_l2)

    print('\n\ntrain metrics layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_train_l2, predicted))
    print('Accuracy = ', accuracy_score(y_train_l2, predicted))
    print('F1 Score = ', f1_score(y_train_l2, predicted))
    print('Precision = ', precision_score(y_train_l2, predicted))
    print('Recall = ', recall_score(y_train_l2, predicted))

    return l2_clf

def dt2():
    print("\nDecision Tree")
    l2_clf = tree.DecisionTreeClassifier()

    start = datetime.now()
    l2_clf.fit(x_train_l2, y_train_l2)
    end = datetime.now() - start

    predicted = l2_clf.predict(x_test_l2)

    print('Metrics for layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l2, predicted))
    print('Accuracy = ', accuracy_score(y_test_l2, predicted))
    print('F1 Score = ', f1_score(y_test_l2, predicted))
    print('Precision = ', precision_score(y_test_l2, predicted))
    print('Recall = ', recall_score(y_test_l2, predicted))
    print('Train time: ', end)

    predicted = l2_clf.predict(x_train_l2)

    print('\n\ntrain metrics layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_train_l2, predicted))
    print('Accuracy = ', accuracy_score(y_train_l2, predicted))
    print('F1 Score = ', f1_score(y_train_l2, predicted))
    print('Precision = ', precision_score(y_train_l2, predicted))
    print('Recall = ', recall_score(y_train_l2, predicted))

    return l2_clf

def gbc2():
    print("\nGradient Boosting Classifier")
    l2_clf = GradientBoostingClassifier()

    start = datetime.now()
    l2_clf.fit(x_train_l2, y_train_l2)
    end = datetime.now() - start

    predicted = l2_clf.predict(x_test_l2)

    print('Metrics for layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_test_l2, predicted))
    print('Accuracy = ', accuracy_score(y_test_l2, predicted))
    print('F1 Score = ', f1_score(y_test_l2, predicted))
    print('Precision = ', precision_score(y_test_l2, predicted))
    print('Recall = ', recall_score(y_test_l2, predicted))
    print('Train time: ', end)

    predicted = l2_clf.predict(x_train_l2)

    print('\n\ntrain metrics layer 2:')
    print('Confusion matrix: [TP FN / FP TN]\n', confusion_matrix(y_train_l2, predicted))
    print('Accuracy = ', accuracy_score(y_train_l2, predicted))
    print('F1 Score = ', f1_score(y_train_l2, predicted))
    print('Precision = ', precision_score(y_train_l2, predicted))
    print('Recall = ', recall_score(y_train_l2, predicted))

    return l2_clf


l1_clf = rf()
l2_clf = svm2()

joblib.dump(l1_clf, "Models/l1_clf.pkl")
joblib.dump(l2_clf, "Models/l2_clf.pkl")
