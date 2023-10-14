# UNSW-NB15 Computer Security Dataset: Analysis through Visualization
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Select the correlated features in the two datasets.
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


# Simulation parameters
pd.set_option("display.max.columns", None)

# division in layer based on attacks
'''
# Splitting the sets in two parts, to mirror the strategy used for the DLHA
# First layer will be trained on 'attacks_for_l1', while second layer on 'attacks_for_l2'
attacks_for_l1 = ['generic', 'exploits', 'fuzzers', 'dos']
attacks_for_l2 = ['reconnaissance', 'analysis', 'backdoor', 'shellcode', 'worms']
'''

# import the train set
df_train = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_training-set.csv', sep=',', header=0)
# put everything in lowercase for train and test sets
df_train = df_train.apply(lambda k: k.astype(str).str.lower())

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

# drop the ID column
del df_train['id']

# Intersected Correlated Feature Selection (ICFS)

# ignore the categorical values for now
categorical_features = ['proto', 'service', 'state', 'attack_cat']  # all non numerical features
features_to_encode = ['proto', 'service', 'state']  # features to one-hot encode later
numerical_features = list(set(df_train.columns) - set(categorical_features))

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

# Now we have 9 different data-frames, each one referred to a specific attack category, all numerical values
# no 'attack_cat' feature. We still have df_train as the original dataset.

# for layer1, PCC between:
# generic - rest
# exploits - rest
# fuzzers - rest
# dos - rest
# select common features for all of these attacks

# start with l1

# features for Generic
rest = compute_set_difference(num_trainset, dos)
rest = compute_set_difference(rest, exploits)
rest = compute_set_difference(rest, fuzzers)
generic_and_rest = pd.concat([rest, generic], axis=0)
generic_all = get_most_correlated_features(generic_and_rest, generic_and_rest['label'], 0.05)
print('Generic: ', num_trainset.shape, generic.shape)
print('Correlated features between Generic attacks and all data: ', len(generic_all))

# features for exploits
rest = compute_set_difference(num_trainset, dos)
rest = compute_set_difference(rest, generic)
rest = compute_set_difference(rest, fuzzers)
exploits_and_rest = pd.concat([rest, exploits], axis=0)
exploits_all = get_most_correlated_features(exploits_and_rest, exploits_and_rest['label'], 0.05)
print('Exploits: ', num_trainset.shape, exploits.shape)
print('Correlated features between Exploit attacks and all data: ', len(exploits_all))
print(exploits_all)

# features for fuzzers
rest = compute_set_difference(num_trainset, dos)
rest = compute_set_difference(rest, generic)
rest = compute_set_difference(rest, exploits)
fuzzers_and_rest = pd.concat([rest, fuzzers], axis=0)
fuzzers_all = get_most_correlated_features(fuzzers_and_rest, fuzzers_and_rest['label'], 0.05)
print('Fuzzers: ', num_trainset.shape, fuzzers.shape)
print('Correlated features between Fuzzers attacks and all data: ', len(fuzzers_all))
print(fuzzers_all)

# features for dos
rest = compute_set_difference(num_trainset, fuzzers)
rest = compute_set_difference(rest, generic)
rest = compute_set_difference(rest, exploits)
dos_and_rest = pd.concat([rest, dos], axis=0)
dos_all = get_most_correlated_features(dos_and_rest, dos_and_rest['label'], 0.05)
print('DoS: ', num_trainset.shape, dos.shape)
print('Correlated features between Generic attacks and all data: ', len(dos_all))
print(dos_all)

# intersect for the optimal features
set_dos = set(dos_all)
set_fuzzers = set(fuzzers_all)
set_exploits = set(exploits_all)
set_generic = set(generic_all)

common_features_l1 = set_generic & set_exploits & set_fuzzers & set_dos
print('Common features to train l1: ', len(common_features_l1), common_features_l1)

# now l2 needs the features to describe the difference between rare attacks and normal traffic
normal = normal[numerical_features]

# features for reconneissance
reconnaissance_and_normal = pd.concat([normal, reconnaissance], axis=0)
reconnaissance_all = get_most_correlated_features(reconnaissance_and_normal, reconnaissance_and_normal['label'], 0.01)
print('Reconnaissance: ', normal.shape, reconnaissance.shape)
print('Correlated features between reconnaissance attacks and normal traffic: ', len(reconnaissance_all))
print(reconnaissance_all)

# features for analysis
analysis_and_normal = pd.concat([normal, analysis], axis=0)
analysis_all = get_most_correlated_features(analysis_and_normal, analysis_and_normal['label'], 0.01)
print('analysis: ', normal.shape, analysis.shape)
print('Correlated features between analysis attacks and normal traffic: ', len(analysis_all))
print(analysis_all)

# features for backdoor
backdoor_and_normal = pd.concat([normal, backdoor], axis=0)
backdoor_all = get_most_correlated_features(backdoor_and_normal, backdoor_and_normal['label'], 0.01)
print('backdoor: ', normal.shape, backdoor.shape)
print('Correlated features between backdoor attacks and normal traffic: ', len(backdoor_all))
print(backdoor_all)

# features for shellcode
shellcode_and_normal = pd.concat([normal, shellcode], axis=0)
shellcode_all = get_most_correlated_features(shellcode_and_normal, shellcode_and_normal['label'], 0.01)
print('shellcode: ', normal.shape, shellcode.shape)
print('Correlated features between shellcode attacks and normal traffic: ', len(shellcode_all))
print(shellcode_all)

# features for worms
worms_and_normal = pd.concat([normal, worms], axis=0)
worms_all = get_most_correlated_features(worms_and_normal, worms_and_normal['label'], 0.01)
print('worms: ', normal.shape, worms.shape)
print('Correlated features between worms attacks and normal traffic: ', len(worms_all))
print(worms_all)

# intersect for the optimal features
set_worms = set(worms_all)
set_shellcode = set(shellcode_all)
set_backdoor = set(backdoor_all)
set_analysis = set(analysis_all)
set_reconnaissance = set(reconnaissance_all)

common_features_l2 = set_worms & set_shellcode & set_backdoor & set_analysis & set_reconnaissance
print('Common features to train l2: ', len(common_features_l2), common_features_l2)

# two different train sets for l1 and l2. Each one must contain the samples from the original dataset according
# to the division in layers, and the features obtained from the ICFS as the only features.
x_train_l1 = df_train[(df_train['attack_cat'] == 'generic') | (df_train['attack_cat'] == 'exploits')
                      | (df_train['attack_cat'] == 'fuzzers') | (df_train['attack_cat'] == 'dos')]

x_train_l1 = x_train_l1[list(common_features_l1)]

x_train_l2 = df_train[(df_train['attack_cat'] == 'worms') | (df_train['attack_cat'] == 'shellcode')
                      | (df_train['attack_cat'] == 'reconnaissance') | (df_train['attack_cat'] == 'backdoor')
                      | (df_train['attack_cat'] == 'analysis')]

x_train_l2 = x_train_l2[list(common_features_l2)]

'''
# MinMax scales the values of df_train between 0-1
scaler1 = MinMaxScaler()

# scaling the numerical values of the train set
df_minmax = scaler1.fit_transform(x_train_l1[numerical_features])
x_train_l1[numerical_features] = pd.DataFrame(df_minmax, columns=x_train_l1[numerical_features].columns)
x_train_l2[numerical_features] = pd.DataFrame(df_minmax, columns=x_train_l2[numerical_features].columns)

# Perform One-hot encoding of the categorical values
ohe = OneHotEncoder(handle_unknown='ignore')
label_enc = ohe.fit_transform(x_train_l1[features_to_encode])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(features_to_encode)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
x_train_l1 = pd.concat([x_train_l1[numerical_features], df_enc],
                     axis=1)  # df_train includes the newly one-hot-encoded columns

label_enc = ohe.fit_transform(x_train_l2[features_to_encode])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(features_to_encode)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
x_train_l2 = pd.concat([x_train_l2[numerical_features], df_enc],
                     axis=1)  # df_train includes the newly one-hot-encoded columns

# sort the columns
x_train_l1 = x_train_l1.sort_index(axis=1)
x_train_l2 = x_train_l2.sort_index(axis=1)

print(x_train_l1)
print(x_train_l2)
'''
