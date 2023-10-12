# Load the pre-trained dos+probe detection model
import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Simulation parameters
pd.set_option("display.max.columns", None)

with open("../../OneDrive/Desktop/new_project/NSL-KDD Outputs/dos_probe_NBC_classifier.pkl", "rb") as f:
    classifier1 = pickle.load(f)

# Load the pre-trained u2r+r2l detection model
with open("../../OneDrive/Desktop/new_project/NSL-KDD Outputs/r2l_u2r_classifier.pkl", "rb") as f:
    classifier2 = pickle.load(f)

# Load the UNSW-NB15 pre-processed test set
with open('UNSW-NB15 Outputs/UNSW-NB15_proprocessed_testset.pkl', "rb") as f:
    x_test, y_test = pickle.load(f)

# Print the sets properties
print(x_test.shape, y_test.shape)
print(x_test)
print(y_test)

# if classifier1 says it's not an anomaly, go to classifier2
y_pred1 = classifier1.predict(x_test)
y_pred2 = classifier2.predict(x_test)

# Evaluate the performance of the two classifiers on the test data
accuracy1 = classifier1.score(x_test, y_test)
accuracy2 = classifier2.score(x_test, y_test)

# Print the accuracy of the two classifiers on the test data
print("Accuracy of classifier 1:", accuracy1)
print("Accuracy of classifier 2:", accuracy2)


