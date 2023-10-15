# UNSW-NB15 Computer Security Dataset: Analysis through Visualization
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler as under_sam
from sklearn.svm import SVC

NEW_ICSF = 0

'''
Structure of the system:
First layer will be trained on 'attacks_for_l1', while second layer on 'attacks_for_l2'
attacks_for_l1 = ['generic', 'exploits', 'fuzzers', 'dos']
attacks_for_l2 = ['reconnaissance', 'analysis', 'backdoor', 'shellcode', 'worms']
'''


# program configuration
def setup():
    pd.set_option("display.max.columns", None)


# Perform Pearson's coefficient with threshold
def get_most_correlated_features(x1, x2, threshold):
    # All elements as numbers
    x1 = x1.apply(pd.to_numeric, errors='ignore')
    x2 = x2.apply(pd.to_numeric, errors='ignore')
    # Calculate the Pearson's correlation coefficient between the two DataFrames.
    corr_matrix = x1.corrwith(x2)
    # print(corr_matrix)
    # Select the features with a high correlation coefficient.
    pcf = corr_matrix.abs().sort_values(ascending=False).index[corr_matrix.abs() > threshold]
    return pcf


# compute all elements of df1 that are not in df2
def compute_set_difference(df1, df2):
    # Create a new DataFrame containing the set difference of the two DataFrames.
    df_diff = df1[~df1.index.isin(df2.index)]
    # Return the DataFrame.
    return df_diff


def read_csv(path):
    dataset = pd.read_csv(path, sep=',', header=0)
    # put everything in lowercase
    dataset = dataset.apply(lambda k: k.astype(str).str.lower())
    # drop the ID column
    del dataset['id']
    return dataset


def perform_icfs(df_train, numerical_features):
    # now ICFS only on the numerical features
    num_trainset = copy.deepcopy(df_train)
    del num_trainset['proto']
    del num_trainset['service']
    del num_trainset['state']

    # let's create a dataframe for each attack category, removing the 'attack_cat' feature
    generic = num_trainset[(num_trainset['attack_cat'] == 'generic')]
    del generic['attack_cat']
    exploits = num_trainset[(num_trainset['attack_cat'] == 'exploits')]
    del exploits['attack_cat']
    fuzzers = num_trainset[(num_trainset['attack_cat'] == 'fuzzers')]
    del fuzzers['attack_cat']
    dos = num_trainset[(num_trainset['attack_cat'] == 'dos')]
    del dos['attack_cat']
    reconnaissance = num_trainset[(num_trainset['attack_cat'] == 'reconnaissance')]
    del reconnaissance['attack_cat']
    analysis = num_trainset[(num_trainset['attack_cat'] == 'analysis')]
    del analysis['attack_cat']
    backdoor = num_trainset[(num_trainset['attack_cat'] == 'backdoor')]
    del backdoor['attack_cat']
    shellcode = num_trainset[(num_trainset['attack_cat'] == 'shellcode')]
    del shellcode['attack_cat']
    worms = num_trainset[(num_trainset['attack_cat'] == 'worms')]
    del worms['attack_cat']
    normal = num_trainset[(num_trainset['attack_cat'] == 'normal')]
    del normal['attack_cat']

    # drop 'attack_cat' for now
    del num_trainset['attack_cat']

    # start with l1

    # features for Generic
    rest = compute_set_difference(num_trainset, dos)
    rest = compute_set_difference(rest, exploits)
    rest = compute_set_difference(rest, fuzzers)
    generic_and_rest = pd.concat([rest, generic], axis=0)
    generic_all = get_most_correlated_features(generic_and_rest, generic_and_rest['label'], 0.05)

    # features for exploits
    rest = compute_set_difference(num_trainset, dos)
    rest = compute_set_difference(rest, generic)
    rest = compute_set_difference(rest, fuzzers)
    exploits_and_rest = pd.concat([rest, exploits], axis=0)
    exploits_all = get_most_correlated_features(exploits_and_rest, exploits_and_rest['label'], 0.05)

    # features for fuzzers
    rest = compute_set_difference(num_trainset, dos)
    rest = compute_set_difference(rest, generic)
    rest = compute_set_difference(rest, exploits)
    fuzzers_and_rest = pd.concat([rest, fuzzers], axis=0)
    fuzzers_all = get_most_correlated_features(fuzzers_and_rest, fuzzers_and_rest['label'], 0.04)

    # features for dos
    rest = compute_set_difference(num_trainset, fuzzers)
    rest = compute_set_difference(rest, generic)
    rest = compute_set_difference(rest, exploits)
    dos_and_rest = pd.concat([rest, dos], axis=0)
    dos_all = get_most_correlated_features(dos_and_rest, dos_and_rest['label'], 0.03)

    # intersect for the optimal features
    set_dos = set(dos_all)
    set_fuzzers = set(fuzzers_all)
    set_exploits = set(exploits_all)
    set_generic = set(generic_all)

    comm_features_l1 = set_generic & set_exploits & set_fuzzers & set_dos

    # now l2 needs the features to describe the difference between rare attacks and normal traffic
    normal = normal[numerical_features]

    # features for reconneissance
    reconnaissance_and_normal = pd.concat([normal, reconnaissance], axis=0)
    reconnaissance_all = get_most_correlated_features(reconnaissance_and_normal, reconnaissance_and_normal['label'],
                                                      0.01)

    # features for analysis
    analysis_and_normal = pd.concat([normal, analysis], axis=0)
    analysis_all = get_most_correlated_features(analysis_and_normal, analysis_and_normal['label'], 0.01)

    # features for backdoor
    backdoor_and_normal = pd.concat([normal, backdoor], axis=0)
    backdoor_all = get_most_correlated_features(backdoor_and_normal, backdoor_and_normal['label'], 0.01)

    # features for shellcode
    shellcode_and_normal = pd.concat([normal, shellcode], axis=0)
    shellcode_all = get_most_correlated_features(shellcode_and_normal, shellcode_and_normal['label'], 0.01)

    # features for worms
    worms_and_normal = pd.concat([normal, worms], axis=0)
    worms_all = get_most_correlated_features(worms_and_normal, worms_and_normal['label'], 0.01)

    # intersect for the optimal features
    set_worms = set(worms_all)
    set_shellcode = set(shellcode_all)
    set_backdoor = set(backdoor_all)
    set_analysis = set(analysis_all)
    set_reconnaissance = set(reconnaissance_all)

    comm_features_l2 = set_worms & set_shellcode & set_backdoor & set_analysis & set_reconnaissance
    # print('Common features to train l2: ', len(common_features_l2), common_features_l2)

    with open('UNSW-NB15 Outputs/UNSW_features_l1.txt', 'w') as g:
        for i, x in enumerate(comm_features_l1):
            if i < len(comm_features_l1) - 1:
                g.write(x + ',')
            else:
                g.write(x)

    # read the common features from file
    with open('UNSW-NB15 Outputs/UNSW_features_l2.txt', 'w') as g:
        for i, x in enumerate(comm_features_l2):
            if i < len(comm_features_l2) - 1:
                g.write(x + ',')
            else:
                g.write(x)


def scaling(dataset, num_col):
    scaler = MinMaxScaler()
    # Create new DataFrames to store the scaled values.
    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset[num_col]), columns=num_col)
    return dataset_scaled


def one_hot_encode(dataset, feat_to_enc):
    ohe = OneHotEncoder(handle_unknown='ignore')
    label_enc = ohe.fit_transform(dataset[feat_to_enc])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(feat_to_enc)
    encoded_dataset = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)

    return encoded_dataset


def perform_pca(dataset):
    pca = PCA(n_components=0.95)
    pca.fit(dataset)
    dataset_pca = pca.transform(dataset)
    return dataset_pca


def perform_under_sampling(train_set, target):
    under_sampler = under_sam(sampling_strategy=1)
    train_set_us, test_set_us = under_sampler.fit_resample(train_set, target)
    return train_set_us, test_set_us


def train_NBC(x_train, y_train):
    l1_calssifier = GaussianNB()  # The classifier selected for DoS+Probe is Naive Bayes Classifier (NBC)
    l1_calssifier.fit(x_train, y_train)  # NBC is trained on the reduced train set and labels
    return l1_calssifier


def train_SVM(x_train, y_train):
    l2_classifier = SVC(C=0.1, gamma=0.01, kernel='rbf')
    l2_classifier.fit(x_train, y_train)
    return l2_classifier


# visualization part commented
'''
# visualize the distribution of attacks in the train set:
attack_cat_counts = df_train['attack_cat'].value_counts().to_frame(name='count')
attack_cat_counts = attack_cat_counts.sort_values(by=['count'], ascending=False)

# Set the size of the figure (adjust width and height as needed)
plt.figure(figsize=(10, 6))
plt.bar(attack_cat_counts.index, attack_cat_counts['count'])
plt.xlabel('Attack Category')
plt.ylabel('Number of Attacks')
plt.title('Distribution of Attacks by Category in train set')
# Rotate the x-axis labels by 45 degrees for better fit
plt.xticks(rotation=90)
plt.tick_params(axis='x', labelsize=8)  # Adjust the fontsize
# Adjust the margins to ensure labels are not cut off
plt.subplots_adjust(top=0.8, bottom=0.2)
plt.savefig('UNSW-NB15 Outputs/attacks_plot_in_train-set.png', dpi=300)

# visualize the distribution of attacks in the test set:
attack_cat_counts = df_test['attack_cat'].value_counts().to_frame(name='count')
attack_cat_counts = attack_cat_counts.sort_values(by=['count'], ascending=False)

# Set the size of the figure (adjust width and height as needed)
plt.figure(figsize=(10, 6))
plt.bar(attack_cat_counts.index, attack_cat_counts['count'])
plt.xlabel('Attack Category')
plt.ylabel('Number of Attacks')
plt.title('Distribution of Attacks by Category in train set')
# Rotate the x-axis labels by 45 degrees for better fit
plt.xticks(rotation=90)
plt.tick_params(axis='x', labelsize=8)  # Adjust the fontsize
# Adjust the margins to ensure labels are not cut off
plt.subplots_adjust(top=0.8, bottom=0.2)
plt.savefig('UNSW-NB15 Outputs/attacks_plot_in_test-set.png', dpi=300)
'''

# write the present features to file
'''
with open('UNSW-NB15 Outputs/Features_from_train_test.txt', 'w') as f:
    for x in df_train.columns:
        f.write(str(x) + '\n')
'''

# Do test and train differ in features?
'''
# let's see if they differ in features
not_present_in_test = []
for x in df_train.columns:
    if x not in df_test.columns:
        not_present_in_test.append(x)
if len(not_present_in_test) == 0:
    print("Test and train have the same features.")
else:
    print('Features in train not present in test: ' + str(not_present_in_test))
'''


def main():
    setup()

    df_train = read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_training-set.csv')
    df_test = read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_testing-set.csv')

    print('train set shape, test set shape: ', df_train.shape, df_test.shape)

    # ignore the categorical values for now
    categorical_features = ['proto', 'service', 'state', 'attack_cat']  # all non numerical features
    features_to_encode = ['proto', 'service', 'state']  # features to one-hot encode later
    numerical_features = list(set(df_train.columns) - set(categorical_features))

    print('numerical features: ', len(numerical_features))
    print('categorical features: ', len(categorical_features))

    # need new ICSF? 0 = no, 1 = yes
    if NEW_ICSF:
        perform_icfs(df_train, numerical_features)

    # read the output features of ICFS from file
    with open('UNSW-NB15 Outputs/UNSW_features_l1.txt', 'r') as f:
        file_contents = f.read()
    common_features_l1 = file_contents.split(',')

    with open('UNSW-NB15 Outputs/UNSW_features_l2.txt', 'r') as f:
        file_contents = f.read()
    common_features_l2 = file_contents.split(',')

    print('features selected for l1: ', len(common_features_l1))
    print('features selected for l2: ', len(common_features_l2))

    # two different train sets for l1 and l2. Each one must contain the samples from the original dataset according
    # to the division in layers, and the features obtained from the ICFS as the only features.
    to_add = list(common_features_l1) + features_to_encode
    x_train_l1 = df_train[list(to_add)]
    x_train_l1 = x_train_l1.sort_index(axis=1)

    x_test_l1 = df_test[list(to_add)]
    x_test_l1 = x_test_l1.sort_index(axis=1)

    x_train_l2 = df_train[(df_train['attack_cat'] == 'worms') | (df_train['attack_cat'] == 'shellcode')
                          | (df_train['attack_cat'] == 'reconnaissance') | (df_train['attack_cat'] == 'backdoor')
                          | (df_train['attack_cat'] == 'analysis') | (df_train['attack_cat'] == 'normal')]

    x_test_l2 = df_test[(df_test['attack_cat'] == 'worms') | (df_test['attack_cat'] == 'shellcode')
                        | (df_test['attack_cat'] == 'reconnaissance') | (df_test['attack_cat'] == 'backdoor')
                        | (df_test['attack_cat'] == 'analysis') | (df_test['attack_cat'] == 'normal')]

    to_add = list(common_features_l2) + features_to_encode
    x_train_l2 = x_train_l2[list(to_add)]
    x_train_l2 = x_train_l2.sort_index(axis=1)

    x_test_l2 = x_test_l2[list(to_add)]
    x_test_l2 = x_test_l2.sort_index(axis=1)

    print('l1 train set shape, l1 test set shape: ', x_train_l1.shape, x_test_l1.shape)
    print('l2 train set shape, l2 test set shape: ', x_train_l2.shape, x_test_l2.shape)

    # MinMax scaling for both sets, only the numerical features present in the sets
    num_l1 = list(set(numerical_features) & set(common_features_l1))
    num_l2 = list(set(numerical_features) & set(common_features_l2))

    # scale the sets
    x_train_l1_scaled = scaling(x_train_l1, num_l1)
    x_train_l2_scaled = scaling(x_train_l2, num_l2)
    x_test_l1_scaled = scaling(x_test_l1, num_l1)
    x_test_l2_scaled = scaling(x_test_l2, num_l2)

    # we now have the numerical features obtained doing the ICFS, now let's handle the categorical ones
    x_train_l1_ohe = one_hot_encode(x_train_l1, features_to_encode)
    x_train_l2_ohe = one_hot_encode(x_train_l2, features_to_encode)
    x_test_l1_ohe = one_hot_encode(x_test_l1, features_to_encode)
    x_test_l2_ohe = one_hot_encode(x_test_l2, features_to_encode)

    # now let's assemble the whole datasets and sort (l1 --> layer 1, l2 --> layer 2)
    l1_train = pd.concat([x_train_l1_scaled, x_train_l1_ohe], axis=1).sort_index(axis=1)
    l2_train = pd.concat([x_train_l2_scaled, x_train_l2_ohe], axis=1).sort_index(axis=1)
    l1_test = pd.concat([x_test_l1_scaled, x_test_l1_ohe], axis=1).sort_index(axis=1)
    l2_test = pd.concat([x_test_l2_scaled, x_test_l2_ohe], axis=1).sort_index(axis=1)

    print('l1 train, l1 test after one-hot encoding: ', l1_train.shape, l1_test.shape)
    print('l2 train, l1 test after one-hot encoding: ', l2_train.shape, l2_test.shape)

    # perform under sampling
    l1_train_us, y_train_l1 = perform_under_sampling(l1_train, l1_train['label'])
    l1_test_us, y_test_l1 = perform_under_sampling(l1_test, l1_test['label'])
    l2_train_us, y_train_l2 = perform_under_sampling(l2_train, l2_train['label'])
    l2_test_us, y_test_l2 = perform_under_sampling(l2_test, l2_test['label'])

    # adjust the features if they mismatch
    for x in l1_train_us.columns:
        if x not in l1_test_us.columns:
            del l1_train_us[x]

    for x in l1_test_us.columns:
        if x not in l1_train_us.columns:
            del l1_test_us[x]

    for x in l2_train_us.columns:
        if x not in l2_test_us.columns:
            del l2_train_us[x]

    for x in l2_test_us.columns:
        if x not in l2_train_us.columns:
            del l2_test_us[x]

    # i saved the output array in y_train variables
    del l1_test_us['label']
    del l2_test_us['label']
    del l1_train_us['label']
    del l2_train_us['label']

    print('l1 train, l1 test after under sampling: ', l1_train_us.shape, l1_test_us.shape)
    print('l2 train, l2 test after under sampling: ', l2_train_us.shape, l2_test_us.shape)

    # pca on all the sets
    l1_train_final = perform_pca(l1_train_us)
    l1_test_final = perform_pca(l1_test_us)
    l2_train_final = perform_pca(l2_train_us)
    l2_test_final = perform_pca(l2_test_us)

    print('l1 train, l1 test, final shape: ', l1_train_final.shape, l1_test_final.shape)
    print('l2 train, l2 test, target final shape: ', l2_train_final.shape, l2_test_final.shape)

    l1_classifier = train_NBC(l1_train_final, y_train_l1)
    predicted_l1 = l1_classifier.predict(l1_test_final)

    l2_classifier = train_SVM(l2_train_final, y_train_l2)
    predicted_l2 = l2_classifier.predict(l2_test_final)


if __name__ == '__main__':
    main()
