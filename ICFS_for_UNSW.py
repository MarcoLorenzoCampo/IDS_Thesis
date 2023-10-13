# UNSW-NB15 Computer Security Dataset: Analysis through Visualization

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Splitting the sets in two parts, to mirror the strategy used for the DLHA
# Normal: Normal, Generic
# DoS: DoS, Worms
# Probe: Reconnaissance
# U2R: Backdoor
# R2L: Exploits, Fuzzers
dos_probe_list = ['dos', 'worms', 'reconnaissance', 'analysis', 'generic']
u2r_r2l_list = ['backdoor', 'exploits', 'fuzzer', 'shellcode']

# import the dataset
df_test = pd.read_csv('UNSW-NB15_datasets/Partial Sets/UNSW_NB15_testing-set.csv', sep=',', header=0)
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
df_test = df_test.apply(lambda k: k.astype(str).str.lower())

# sort the columns
df_train = df_train.sort_index(axis=1)  # sort the columns
df_test = df_test.sort_index(axis=1)  # sort the columns

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
to_append_test_atk = df_test['attack_cat']
to_append_train_lbl = df_train['label']
to_append_test_lbl = df_test['label']

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

print(df_train.shape)
print()
print(df_train.head())

# now do the same for the test set
# scaling the numerical values of the train set
df_minmax = scaler1.fit_transform(df_test[num_col])
df_test[num_col] = pd.DataFrame(df_minmax, columns=df_test[num_col].columns)

# Perform One-hot encoding of the categorical values
label_enc = ohe.fit_transform(df_test[to_encode])
label_enc.toarray()
new_labels = ohe.get_feature_names_out(to_encode)
df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
df_test = pd.concat([df_test[num_col], df_enc, to_append_test_atk, to_append_test_lbl],
                     axis=1)  # df_train includes the newly one-hot-encoded columns

print(df_test.shape)
print()
print(df_test.head())