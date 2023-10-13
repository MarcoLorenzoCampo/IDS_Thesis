import pandas as pd
import matplotlib.pyplot as plt

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

# let's see if they differ in features
not_present_in_test = []
for x in df_train.columns:
    if x not in df_test.columns:
        not_present_in_test.append(x)
if len(not_present_in_test) == 0:
    print("Test and train have the same features.")
else:
    print('Features in train not present in test: ' + str(not_present_in_test))

# put everything in lowercase for train and test sets
df_train = df_train.apply(lambda k: k.astype(str).str.lower())
df_test = df_test.apply(lambda k: k.astype(str).str.lower())

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