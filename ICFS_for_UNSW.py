# UNSW-NB15 Computer Security Dataset: Analysis through Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Simulation parameters
pd.set_option("display.max.columns", None)

# Splitting the sets in two parts, to mirror the strategy used for the DLHA
# First layer will be trained on 'attacks_for_l1', while second layer on 'attacks_for_l2'
attacks_for_l1 = ['generic', 'exploits', 'fuzzers', 'dos']
attacks_for_l2 = ['reconnaissance', 'analysis', 'backdoor', 'shellcode', 'worms']

# import the dataset
# df_test = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_testing-set.csv', sep=',', header=0)
df_train = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_training-set.csv', sep=',', header=0)

# write the present features to file
with open('UNSW-NB15 Outputs/Features_from_train_test.txt', 'w') as f:
    for x in df_train.columns:
        f.write(str(x) + '\n')

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

# put everything in lowercase for train and test sets
df_train = df_train.apply(lambda k: k.astype(str).str.lower())
# df_test = df_test.apply(lambda k: k.astype(str).str.lower())

# drop the ID column
del df_train['id']
# del df_test['id']

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

# The following is applied to train set and later to test set
# ignore the categorical values for now
non_numerical = ['proto', 'service', 'state', 'attack_cat', 'label']
to_encode = ['proto', 'service', 'state']
num_col = list(set(df_train.columns) - set(non_numerical))
to_append_train_atk = df_train['attack_cat']
# to_append_test_atk = df_test['attack_cat']
to_append_train_lbl = df_train['label']
# to_append_test_lbl = df_test['label']

# MinMax scales the values of df_train between 0-1
scaler1 = MinMaxScaler()

# scaling the numerical values of the train set
df_minmax = scaler1.fit_transform(df_train[num_col])
df_train[num_col] = pd.DataFrame(df_minmax, columns=df_train[num_col].columns)

# Perform One-hot encoding of the categorical values
ohe = OneHotEncoder(handle_unknown='ignore')
label_enc = ohe.fit_transform(df_train[to_encode])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(to_encode)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
df_train = pd.concat([df_train[num_col], df_enc, to_append_train_atk, to_append_train_lbl],
                     axis=1)  # df_train includes the newly one-hot-encoded columns

# sort the columns
df_train = df_train.sort_index(axis=1)  # sort the columns
# df_test = df_test.sort_index(axis=1)  # sort the columns

# now let's perform Intersected Correlated Feature Selection
generic = df_train[(df_train['attack_cat'] == 'generic')]
del generic['attack_cat']
exploits = df_train[(df_train['attack_cat'] == 'exploits')]
del exploits['attack_cat']
fuzzers = df_train[(df_train['attack_cat'] == 'fuzzers')]
del fuzzers['attack_cat']
dos = df_train[(df_train['attack_cat'] == 'dos')]
del dos['attack_cat']
reconnaissance = df_train[(df_train['attack_cat'] == 'reconnaissance')]
del reconnaissance['attack_cat']
analysis = df_train[(df_train['attack_cat'] == 'analysis')]
del analysis['attack_cat']
backdoor = df_train[(df_train['attack_cat'] == 'backdoor')]
del backdoor['attack_cat']
shellcode = df_train[(df_train['attack_cat'] == 'shellcode')]
del shellcode['attack_cat']
worms = df_train[(df_train['attack_cat'] == 'worms')]
del worms['attack_cat']

# drop 'attack_cat' for now
del df_train['attack_cat']


# for layer1, PCC between:
# generic - rest
# exploits - rest
# fuzzers - rest
# dos - rest
# select common features for all of these attacks

# Select the correlated features in the two datasets.
def get_most_correlated_features(x1, x2, n=10):
    # Convert all datasets to numbers
    x1 = x1.apply(pd.to_numeric, errors='ignore')
    x2 = x2.apply(pd.to_numeric, errors='ignore')

    # Calculate the correlation matrix between the two DataFrames.
    corr_matrix = x1.corrwith(x2)

    # Calculate the correlation matrix between the two DataFrames.
    corr_matrix = x1.corrwith(x2)

    # Select the features that are most correlated between the two DataFrames.
    most_correlated_features = corr_matrix.abs().sort_values(ascending=False).index[:n]

    # Return the most important features.
    return most_correlated_features


# compute all elements of df1 that are not in df2
def compute_set_difference(df1, df2):
    # Create a new DataFrame containing the set difference of the two DataFrames.
    df_diff = df1[~df1.index.isin(df2.index)]

    # Return the DataFrame.
    return df_diff


# start with l1
rest = compute_set_difference(df_train, generic)
generic_all = get_most_correlated_features(rest, generic)
print('Generic: ', df_train.shape, generic.shape)
print('Correlated features between Generic attacks and all data: ', len(generic_all))
print(generic_all)

rest = compute_set_difference(df_train, exploits)
exploits_all = get_most_correlated_features(rest, exploits)
print('Exploits: ', df_train.shape, exploits.shape)
print('Correlated features between Exploit attacks and all data: ', len(exploits_all))
print(exploits_all)

rest = compute_set_difference(df_train, fuzzers)
fuzzers_all = get_most_correlated_features(rest, fuzzers)
print('Fuzzers: ', df_train.shape, fuzzers.shape)
print('Correlated features between Fuzzers attacks and all data: ', len(fuzzers_all))
print(fuzzers_all)

rest = compute_set_difference(df_train, dos)
dos_all = get_most_correlated_features(rest, dos)
print('DoS: ', df_train.shape, dos.shape)
print('Correlated features between Generic attacks and all data: ', len(dos_all))
print(dos_all)

# intersect for the optimal features
set_dos = set(dos_all)
set_fuzzers = set(fuzzers_all)
set_exploits = set(exploits_all)
set_generic = set(generic_all)

common_features_l1 = set_generic & set_exploits & set_fuzzers & set_dos
print('Common features to train l1: ', len(common_features_l1), common_features_l1)

# now for l2
rest = compute_set_difference(df_train, reconnaissance)
reconnaissance_all = get_most_correlated_features(rest, reconnaissance)
print('Reconnaissance: ', df_train.shape, reconnaissance.shape)
print('Correlated features between reconnaissance attacks and all data: ', len(reconnaissance_all))
print(reconnaissance_all)

rest = compute_set_difference(df_train, analysis)
analysis_all = get_most_correlated_features(rest, analysis)
print('analysis: ', df_train.shape, analysis.shape)
print('Correlated features between analysis attacks and all data: ', len(analysis_all))
print(analysis_all)

rest = compute_set_difference(df_train, backdoor)
backdoor_all = get_most_correlated_features(rest, backdoor)
print('backdoor: ', df_train.shape, backdoor.shape)
print('Correlated features between backdoor attacks and all data: ', len(backdoor_all))
print(backdoor_all)

rest = compute_set_difference(df_train, shellcode)
shellcode_all = get_most_correlated_features(rest, shellcode)
print('shellcode: ', df_train.shape, shellcode.shape)
print('Correlated features between shellcode attacks and all data: ', len(shellcode_all))
print(shellcode_all)

rest = compute_set_difference(df_train, worms)
worms_all = get_most_correlated_features(rest, worms)
print('worms: ', df_train.shape, worms.shape)
print('Correlated features between worms attacks and all data: ', len(worms_all))
print(worms_all)

# intersect for the optimal features
set_worms = set(worms_all)
set_shellcode = set(shellcode_all)
set_backdoor = set(backdoor_all)
set_analysis = set(analysis_all)
set_reconnaissance = set(reconnaissance_all)

common_features_l2 = set_worms & set_shellcode & set_backdoor & set_analysis & set_reconnaissance
print('Common features to train l2: ', len(common_features_l2), common_features_l2)