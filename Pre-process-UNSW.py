import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler as under_sam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Testing the model with a different test set
pd.set_option("display.max.columns", 10)

# Loading train and test set
df_test = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_testing-set.csv', sep=',', header=0)
df_train = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_training-set.csv', sep=',', header=0)

# put everything in lowercase for both sets
df_train = df_train.apply(lambda x: x.astype(str).str.lower())
df_test = df_test.apply(lambda x: x.astype(str).str.lower())

labels_train = df_train.iloc[:, -1]  # List of output test labels for each sample
attacks_train = df_train.iloc[:, -2]  # List of test attacks types
labels_test = df_test.iloc[:, -1]  # List of output test labels for each sample
attacks_test = df_test.iloc[:, -2]  # List of test attacks types
df_train = df_train.drop(columns=df_test.columns[0])  # Drop the ID feature for the train set
df_test = df_test.drop(columns=df_test.columns[0])  # Drop the ID feature for the test set

with open('UNSW-NB15 Outputs/Results.txt', 'w') as output_file:
    output_file.write('Results for the training and testing of DLHA:  Double-Layered Hybrid Approach'
                      ', an Anomaly-Based Intrusion Detection System (UNSBW-NB15 dataset)\n')
    output_file.write('\nShape of the training set: (#samples, #features) =' + str(df_train.shape))
    output_file.write('\nShape of the testing set: (#samples, #features) =' + str(df_test.shape))

# Splitting the sets in two parts, to mirror the strategy used for the DLHA
# Normal: Normal, Generic
# DoS: DoS, Worms
# Probe: Reconnaissance
# U2R: Backdoor
# R2L: Exploits, Fuzzers
dos_probe_list = ['dos', 'worms', 'reconnaissance']
u2r_r2l_list = ['backdoor', 'exploits', 'fuzzer', 'shellcode']

# set the target variables
y_train = np.array([1 if (x in dos_probe_list or x in u2r_r2l_list) else 0 for x in df_train['label']])
y_test = np.array([1 if (x in dos_probe_list or x in u2r_r2l_list) else 0 for x in df_test['label']])

# remove the labels and reset the index
df_train = df_train.drop(['label'], axis=1)
df_train = df_train.reset_index().drop(['index'], axis=1)
df_test = df_test.drop(['label'], axis=1)
df_test = df_test.reset_index().drop(['index'], axis=1)

# ignore the categorical values for now
cat_col = ['proto', 'service', 'state']
num_col = list(set(df_train.columns) - set(cat_col))

# subset of features from the original dataset, remove the attack column, will be added later
X_train = df_train[num_col]
X_train = X_train.drop('attack_cat', axis=1)
X_test = df_test[num_col]
X_test = X_test.drop('attack_cat', axis=1)

# MinMax scales the values of X_train and X_test between 0-1
scaler1 = MinMaxScaler()

# scaling the train set
df_minmax = scaler1.fit_transform(X_train)
X_train = pd.DataFrame(df_minmax, columns=X_train.columns)

# scaling the test set
df_minmax = scaler1.fit_transform(X_test)
X_test = pd.DataFrame(df_minmax, columns=X_test.columns)

# Perform One-hot encoding of the categorical values
ohe = OneHotEncoder(handle_unknown='ignore')

# for the train set
label_enc = ohe.fit_transform(df_train[cat_col])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(cat_col)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
X_train = pd.concat([X_train, df_enc], axis=1)  # X_train includes now the newly one-hot-encoded columns
X_train = pd.concat([X_train, attacks_train], axis=1)  # add the attack column back
X_train = X_train.sort_index(axis=1)  # sort the columns

# for the test set
label_enc = ohe.fit_transform(df_test[cat_col])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(cat_col)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
X_test = pd.concat([X_test, df_enc], axis=1)  # X_test includes now the newly one-hot-encoded columns
X_test = pd.concat([X_test, attacks_test], axis=1)  # add the attack column back

print('\nAfter OHE: ', X_train.shape, X_test.shape)

# check the features
not_present_in_test = []
for x in X_train.columns:
    if x not in X_test.columns:
        not_present_in_test.append(x)
print('Features: ' + str(not_present_in_test) + ' are OHE features not present in the test-set. Remove them manually.')

# check the features
not_present_in_train = []
for x in X_test.columns:
    if x not in X_train.columns:
        not_present_in_train.append(x)
print('Features: ' + str(not_present_in_train) + ' are OHE features not present in the train-set. '
                                                 'Remove them manually.')

# add the missing features to the test set as all zeros, they are one hot encoded
data = np.zeros((82332, len(not_present_in_test)))
df = pd.DataFrame(data, columns=not_present_in_test)
X_test = pd.concat([X_test, df], axis=1)
X_test = X_test.sort_index(axis=1)

# add the missing features to the train set as all zeros, they are one hot encoded
data = np.zeros((175341, len(not_present_in_train)))
df = pd.DataFrame(data, columns=not_present_in_train)
X_train = pd.concat([X_train, df], axis=1)
X_train = X_train.sort_index(axis=1)

# Writing to file the results of the preprocessing
with open('UNSW-NB15 Outputs/Results.txt', 'a') as output_file:
    output_file.write("\nProperties of training set and test set for NBC:\n")
    output_file.write('\nShape of the training data features: (#samples, #features) =' + str(X_train.shape))
    output_file.write('\nShape of the training data labels: (#samples,) =' + str(y_train.shape))
    output_file.write('\nShape of the test data features: (#samples, #features) =' + str(X_test.shape))
    output_file.write('\nShape of the test data labels: (#samples,) =' + str(y_test.shape))

# We have 1 training set and 1 test set. We want 2 split according to their attack category
# Splitting the sets in two parts, to mirror the strategy used for the DLHA
# Normal: Normal, Generic
# DoS: DoS, Worms
# Probe: Reconnaissance
# U2R: Backdoor
# R2L: Exploits, Fuzzers
X_train_l1 = X_train
X_train_l2 = X_train[(X_train['attack_cat'] == 'normal') | (X_train['attack_cat'] == 'generic') |
                     (X_train['attack_cat'] == 'backdoor') | (X_train['attack_cat'] == 'exploits') |
                     (X_train['attack_cat'] == 'fuzzer') | (X_train['attack_cat'] == 'shellcode')]

X_test_l1 = X_test
X_test_l2 = X_test[(X_test['attack_cat'] == 'normal') | (X_test['attack_cat'] == 'generic') |
                   (X_test['attack_cat'] == 'backdoor') | (X_test['attack_cat'] == 'exploits') |
                   (X_test['attack_cat'] == 'fuzzer') | (X_train['attack_cat'] == 'shellcode')]

# set the target variables
y_train_l1 = np.array([1 if (x in dos_probe_list or x in u2r_r2l_list) else 0 for x in X_train_l1['attack_cat']])
y_test_l1 = np.array([1 if (x in dos_probe_list or x in u2r_r2l_list) else 0 for x in X_test_l1['attack_cat']])

y_train_l2 = np.array([1 if (x in u2r_r2l_list) else 0 for x in X_train_l2['attack_cat']])
y_test_l2 = np.array([1 if (x in u2r_r2l_list) else 0 for x in X_test_l2['attack_cat']])

# define the under sampler, to reduce the disparity of classes
under_sampler = under_sam(sampling_strategy=1)

# balancing the train set for l1. Contains all the attacks/normal traffic.
X_train_l1, y_train_l1 = under_sampler.fit_resample(X_train_l1, y_train_l1)
# balancing the test set for l1. Contains all the attacks/normal traffic.
X_test_l1, y_test_l1 = under_sampler.fit_resample(X_test_l1, y_test_l1)

# balancing the train set for l2. Contains only the attacks pertaining u2r and r2l.
X_train_l2, y_train_l2 = under_sampler.fit_resample(X_train_l2, y_train_l2)
# balancing the test set for l2. Contains only the attacks pertaining u2r and r2l.
X_test_l2, y_test_l2 = under_sampler.fit_resample(X_test_l2, y_test_l2)

print('\nBefore PCA:')
print('Shape of the train set for layer1: ', X_train_l1.shape, ' - Target variable: ', len(y_train_l1))
print('Shape of the train set for layer2: ', X_train_l2.shape, ' - Target variable: ', len(y_train_l2))
print('Shape of the test set for layer1: ', X_test_l1.shape, ' - Target variable: ', len(y_test_l1))
print('Shape of the test set for layer2: ', X_test_l2.shape, ' - Target variable: ', len(y_test_l2))

# now we can finally remove the 'attack_cat' column from all sets
del X_train_l1['attack_cat']
del X_train_l2['attack_cat']
del X_test_l1['attack_cat']
del X_test_l2['attack_cat']

# Now that all the sets have been set up, we can reason on the features
# starting with the features for l1
pca_dos_probe = PCA(n_components=22)  # features selected can explain >95% of the variance
X_train_dos_probe = pca_dos_probe.fit_transform(X_train_l1)  # Reduce the dimensionality of X_train
X_test_dos_probe = pca_dos_probe.fit_transform(X_test_l1)  # Reduce the dimensionality of X_test

# features for l2
pca_dos_probe = PCA(n_components=20)  # features selected can explain >95% of the variance
X_train_r2l_u2r = pca_dos_probe.fit_transform(X_train_l2)  # Reduce the dimensionality of X_train
X_test_r2l_u2r = pca_dos_probe.fit_transform(X_test_l2)  # Reduce the dimensionality of X_test

print('\nAfter PCA:')
print('Shape of the train set for layer1: ', X_train_dos_probe.shape, ' - Target variable: ', len(y_train_l1))
print('Shape of the train set for layer2: ', X_train_r2l_u2r.shape, ' - Target variable: ', len(y_train_l2))
print('Shape of the test set for layer1: ', X_test_dos_probe.shape, ' - Target variable: ', len(y_test_l1))
print('Shape of the test set for layer2: ', X_test_r2l_u2r.shape, ' - Target variable: ', len(y_test_l2))

dos_probe_classifier = GaussianNB()  # The classifier selected for DoS+Probe is Naive Bayes Classifier (NBC)
dos_probe_classifier.fit(X_train_dos_probe, y_train_l1)  # NBC is trained on the reduced train set and labels
predicted_l1 = dos_probe_classifier.predict(X_test_dos_probe)  # Evaluates the model on the test data

# Support Vector Classifier with parameters C, gamma and kernel function 'radial basis function'
r2l_u2r_classifier = SVC(C=0.1, gamma=0.01, kernel='rbf')
r2l_u2r_classifier.fit(X_train_r2l_u2r, y_train_l2)
predicted_l2 = r2l_u2r_classifier.predict(X_test_r2l_u2r)

# compute the parameters
accuracy = accuracy_score(y_test_l1, predicted_l1)
# compute the confusion matrix
confusion_matrix1 = confusion_matrix(y_test_l1, predicted_l1)
print('\nAccuracy for l1: ', accuracy)
print('Confusion matrix for l1: ', confusion_matrix1)

# for l1
accuracy = accuracy_score(y_test_l2, predicted_l2)
# compute the confusion matrix
confusion_matrix2 = confusion_matrix(y_test_l2, predicted_l2)
print('\nAccuracy for l2: ', accuracy)
print('Confusion matrix for l2: ', confusion_matrix2)

'''
# Write the output on the text file
with open('UNSW-NB15 Outputs/Results.txt', 'a') as output_file:
    output_file.write("\n\n\nTesting results:\n")

    output_file.write('\nLayer1:')
    output_file.write('\n\nTotal samples: ' + str(X_train_dos_probe.shape))
    output_file.write('\nDetected R2L: ' + str(result[r2l_index].sum()))
    output_file.write('\nR2L results correctness: ' + str(result[r2l_index].sum() / result[r2l_index].shape[0]*100) + '%')

    output_file.write('\n\nTotal U2R samples: ' + str(result[u2r_index].shape[0]))
    output_file.write('\nDetected U2R: ' + str(result[u2r_index].sum()))
    output_file.write('\nU2R results correctness: ' + (str(result[u2r_index].sum() / result[u2r_index].shape[0]*100)) + '%')
'''