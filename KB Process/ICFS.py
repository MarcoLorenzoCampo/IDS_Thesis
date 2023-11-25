import copy

import numpy as np
import pandas as pd


def __pearson_correlated_features(x, y, threshold):
    y['target'] = y['target'].astype(int)

    for p in x.columns:
        x[p] = x[p].astype(float)

    # Ensure y is a DataFrame for consistency
    if isinstance(y, pd.Series):
        y = pd.DataFrame(y, columns=['target'])

    # Calculate the Pearson's correlation coefficients between features and the target variable(s)
    corr_matrix = x.corrwith(y['target'])

    # Select features with correlations above the threshold
    selected_features = x.columns[corr_matrix.abs() > threshold].tolist()

    return selected_features


def __compute_set_difference(df1, df2):
    # Create a new DataFrame containing the set difference of the two DataFrames.
    df_diff = df1[~df1.index.isin(df2.index)]
    # Return the DataFrame.
    return df_diff


def perform_icfs(x_train):
    # now ICFS only on the numerical features
    num_train = copy.deepcopy(x_train)
    del num_train['protocol_type']
    del num_train['service']
    del num_train['flag']

    target = pd.DataFrame()
    target['target'] = np.array([1 if x != 'normal' else 0 for x in num_train['label']])
    num_train = pd.concat([num_train, target], axis=1)

    # These are how attacks are categorized in the trainset
    dos_list = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    probe_list = ['ipsweep', 'portsweep', 'satan', 'nmap']
    u2r_list = ['loadmodule', 'perl', 'rootkit', 'buffer_overflow']
    r2l_list = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
    normal = ['normal']

    # useful sub-sets
    x_normal = num_train[num_train['label'].isin(normal)]
    x_u2r = num_train[num_train['label'].isin(u2r_list)]
    x_r2l = num_train[num_train['label'].isin(r2l_list)]
    x_dos = num_train[num_train['label'].isin(dos_list)]
    x_probe = num_train[num_train['label'].isin(probe_list)]

    # start the ICFS with l1

    # features for dos
    dos = copy.deepcopy(num_train)
    del dos['target']
    y = np.array([1 if x in dos_list else 0 for x in dos['label']])
    y_dos = pd.DataFrame(y, columns=['target'])
    del dos['label']
    dos_all = __pearson_correlated_features(dos, y_dos, 0.1)
    print(dos_all)

    # features for probe
    probe = copy.deepcopy(num_train)
    del probe['target']
    y = np.array([1 if x in probe_list else 0 for x in probe['label']])
    y_probe = pd.DataFrame(y, columns=['target'])
    del probe['label']
    probe_all = __pearson_correlated_features(probe, y_probe, 0.1)
    print(probe_all)

    # intersect for the optimal features
    set_dos = set(dos_all)
    set_probe = set(probe_all)

    comm_features_l1 = set_probe & set_dos

    print('common features to train l1: ', comm_features_l1)

    # now l2 needs the features to describe the difference between rare attacks and normal traffic

    # features for u2r
    u2r = pd.concat([x_u2r, x_normal], axis=0)
    del u2r['target']
    y = np.array([1 if x in u2r_list else 0 for x in u2r['label']])
    y_u2r = pd.DataFrame(y, columns=['target'])
    del u2r['label']
    u2r_all = __pearson_correlated_features(u2r, y_u2r, 0.01)
    print(u2r_all)

    # features for r2l
    r2l = pd.concat([x_r2l, x_normal], axis=0)
    del r2l['target']
    y = np.array([1 if x in r2l_list else 0 for x in r2l['label']])
    y_r2l = pd.DataFrame(y, columns=['target'])
    del r2l['label']
    r2l_all = __pearson_correlated_features(r2l, y_r2l, 0.01)
    print(r2l_all)

    # intersect for the optimal features
    set_r2l = set(r2l_all)
    set_u2r = set(u2r_all)

    comm_features_l2 = set_r2l & set_u2r
    # print('Common features to train l2: ', len(common_features_l2), common_features_l2)

    with open('KB Process/Required Files/test_l1.txt', 'w') as g:
        for a, x in enumerate(comm_features_l1):
            if a < len(comm_features_l1) - 1:
                g.write(x + ',' + '\n')
            else:
                g.write(x)

    # read the common features from file
    with open('KB Process/Required Files/test_l2.txt', 'w') as g:
        for a, x in enumerate(comm_features_l2):
            if a < len(comm_features_l2) - 1:
                g.write(x + ',' + '\n')
            else:
                g.write(x)
