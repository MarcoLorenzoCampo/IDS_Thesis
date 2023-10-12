import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

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
u2r_r2l_list = ['backdoor', 'exploits', 'fuzzer']

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
print('Features: ' + str(not_present_in_test) + ' are OHE features not present in the test-set. Add them manually.')

# check the features
not_present_in_train = []
for x in X_test.columns:
    if x not in X_train.columns:
        not_present_in_train.append(x)
print('Features: ' + str(not_present_in_train) + ' are OHE features not present in the train-set. Add them manually.')

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
                     (X_train['attack_cat'] == 'fuzzer')]

X_test_l1 = X_test
X_test_l2 = X_test[(X_test['attack_cat'] == 'normal') | (X_test['attack_cat'] == 'generic') |
                     (X_test['attack_cat'] == 'backdoor') | (X_test['attack_cat'] == 'exploits') |
                     (X_test['attack_cat'] == 'fuzzer')]

# now that we split the sets, we can remove the 'attack_cat' feature


print('Shape of the train set for layer1: ', X_train_l1.shape)
print('Shape of the train set for layer2: ', X_train_l2.shape)
print('Shape of the test set for layer1: ', X_test_l1.shape)
print('Shape of the test set for layer2: ', X_test_l2.shape)

# Now that the data pre-processing is done, let's do feature selection
pca_dos_probe = PCA(n_components=0.95)  # features selected can explain >95% of the variance
X_train_dos_probe = pca_dos_probe.fit_transform(X_train)  # Reduce the dimensionality of X_train
X_test_dos_probe = pca_dos_probe.transform(X_test)  # Reduce the dimensionality of X_test
