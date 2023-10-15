import copy
import pickle

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler as under_sam
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# Flags for program behavior, 0 = don't perform, 1 = perform
EXPORT_DATASETS = 0
EXPORT_MODELS = 0


def setup():
    pd.set_option("display.max.columns", None)


def scaling(dataset):
    scaler = MinMaxScaler()
    # Create new DataFrames to store the scaled values.
    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    return dataset_scaled


# Perform Pearson's coefficient with threshold
def get_most_correlated_features(x, y, threshold):
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


# compute all elements of df1 that are not in df2
def compute_set_difference(df1, df2):
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
    normal_list = ['normal']

    # useful sub-sets
    x_normal = num_train[num_train['label'].isin(normal_list)]
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
    dos_all = get_most_correlated_features(dos, y_dos, 0.1)
    print(dos_all)

    # features for probe
    probe = copy.deepcopy(num_train)
    del probe['target']
    y = np.array([1 if x in probe_list else 0 for x in probe['label']])
    y_probe = pd.DataFrame(y, columns=['target'])
    del probe['label']
    probe_all = get_most_correlated_features(probe, y_probe, 0.1)
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
    u2r_all = get_most_correlated_features(u2r, y_u2r, 0.01)
    print(u2r_all)

    # features for r2l
    r2l = pd.concat([x_r2l, x_normal], axis=0)
    del r2l['target']
    y = np.array([1 if x in r2l_list else 0 for x in r2l['label']])
    y_r2l = pd.DataFrame(y, columns=['target'])
    del r2l['label']
    r2l_all = get_most_correlated_features(r2l, y_r2l, 0.01)
    print(r2l_all)

    # intersect for the optimal features
    set_r2l = set(r2l_all)
    set_u2r = set(u2r_all)

    comm_features_l2 = set_r2l & set_u2r
    # print('Common features to train l2: ', len(common_features_l2), common_features_l2)

    with open('NSL-KDD Outputs/NSL_features_l1.txt', 'w') as g:
        for a, x in enumerate(comm_features_l1):
            if a < len(comm_features_l1) - 1:
                g.write(x + ',' + '\n')
            else:
                g.write(x)

    # read the common features from file
    with open('NSL-KDD Outputs/NSL_features_l2.txt', 'w') as g:
        for a, x in enumerate(comm_features_l2):
            if a < len(comm_features_l2) - 1:
                g.write(x + ',' + '\n')
            else:
                g.write(x)


def main():
    setup()

    # loading feature names and value categories
    titles = pd.read_csv('NSL-KDD_datasets/Field Names.csv', header=None)
    label = pd.Series(['label'], index=[41])
    titles = pd.concat([titles[0], label])

    # loading the train set
    df_train = pd.read_csv('NSL-KDD_datasets/KDDTrain+.txt', sep=",", header=None)
    df_train = df_train[df_train.columns[:-1]]  # Drops the first column
    df_train.columns = titles.to_list()  # Gets a list like this: <index> <feature name>
    df_train = df_train.drop(['num_outbound_cmds'], axis=1)
    df_train_original = df_train
    df_train_original = df_train_original.apply(pd.to_numeric, errors='ignore')
    df_train_original = df_train_original.apply(lambda k: k.astype(str).str.lower())

    # load the test set
    df_test = pd.read_csv('NSL-KDD_datasets/KDDTest+.txt', sep=",", header=None)
    df_test = df_test[df_test.columns[:-1]]
    df_test.columns = titles.to_list()
    df_test = df_test.drop(['num_outbound_cmds'], axis=1)
    df_test_original = df_test
    df_test_original = df_test_original.apply(pd.to_numeric, errors='ignore')
    df_test_original = df_test_original.apply(lambda k: k.astype(str).str.lower())

    # load the features obtained with ICFS
    with open('NSL-KDD Outputs/NSL_features_l1.txt', 'r') as f:
        file_contents = f.read()
    common_features_l1 = file_contents.split(',')

    with open('NSL-KDD Outputs/NSL_features_l2.txt', 'r') as f:
        file_contents = f.read()
    common_features_l2 = file_contents.split(',')

    # dos + probe classifier
    df_train = copy.deepcopy(df_train_original)  # Produces copies of the original data sets
    df_test = copy.deepcopy(df_test_original)

    # perform_icfs(df_train)

    # List of DoS+Probe attacks
    dos_probe_list = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                      'udpstorm', 'worm', 'ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']

    y_train = np.array([1 if x in dos_probe_list else 0 for x in df_train['label']])

    df_train = df_train.drop(['label'], axis=1)
    df_train = df_train.reset_index().drop(['index'], axis=1)

    # Features obtained with the ICFS for DoS and Probe
    X_train = df_train[['logged_in', 'count', 'serror_rate', 'rerror_rate',
                        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                        'srv_diff_host_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                        'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate',
                        'dst_host_rerror_rate']]

    scaler1 = MinMaxScaler()  # MinMax scales the values of X_train between 0-1
    df_minmax = scaler1.fit_transform(X_train)
    X_train = pd.DataFrame(df_minmax, columns=X_train.columns)

    ohe = OneHotEncoder(handle_unknown='ignore')  # Perform One-hot encoding
    label_enc = ohe.fit_transform(df_train.iloc[:, 1:4])
    label_enc.toarray()

    # Get the names of the one-hot-encoded columns
    new_labels = ohe.get_feature_names_out(['protocol_type', 'service', 'flag'])

    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_train = pd.concat([X_train, df_enc], axis=1)  # X_train includes now the newly one-hot-encoded columns

    y_test = np.array([1 if x in dos_probe_list else 0 for x in df_test['label']])  # Same for the test set

    df_test = df_test.drop(['label'], axis=1)
    df_test = df_test.reset_index().drop(['index'], axis=1)

    # Same features selected for the train set
    X_test = df_test[['logged_in', 'count', 'serror_rate', 'rerror_rate',
                      'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                      'srv_diff_host_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate',
                      'dst_host_rerror_rate']]

    df_minmax = scaler1.transform(X_test)
    X_test = pd.DataFrame(df_minmax, columns=X_test.columns)

    label_enc = ohe.transform(df_test.iloc[:, 1:4])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(['protocol_type', 'service', 'flag'])
    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_test = pd.concat([X_test, df_enc], axis=1)

    with open('NSL-KDD Outputs/Results.txt', 'w') as output_file:
        output_file.write('Results for the training and testing of DLHA:  Double-Layered Hybrid Approach'
                          ', an Anomaly-Based Intrusion Detection System\n')
        output_file.write("\nProperties of training set and test set for NBC:\n")
        output_file.write('\nShape of the training data features: (#samples, #features) =' + str(X_train.shape))
        output_file.write('\nShape of the training data labels: (#samples,) =' + str(y_train.shape))
        output_file.write('\nShape of the test data features: (#samples, #features) =' + str(X_test.shape))
        output_file.write('\nShape of the test data labels: (#samples,) =' + str(y_test.shape))

    if EXPORT_DATASETS:
        # Export the datasets used for the NBC
        X_train.to_csv('Updated Datasets/X_train_NBC.csv', index=False)
        y_train_df = pd.DataFrame({'label': y_train})
        y_train_df.to_csv('Updated Datasets/y_train_NBC.csv', index=False)

        X_test.to_csv('Updated Datasets/X_test_NBC.csv', index=False)
        y_test_df = pd.DataFrame({'label': y_test})
        y_test_df.to_csv('Updated Datasets/y_test_NBC.csv', index=False)

    # Now that the data pre-processing is done, let's do feature selection
    pca_dos_probe = PCA(n_components=0.95)  # features selected can explain >95% of the variance
    X_train_dos_probe = pca_dos_probe.fit_transform(X_train)  # Reduce the dimensionality of X_train
    X_test_dos_probe = pca_dos_probe.transform(X_test)  # Reduce the dimensionality of X_test

    dos_probe_classifier = GaussianNB()  # The classifier selected for DoS+Probe is Naive Bayes Classifier (NBC)
    dos_probe_classifier.fit(X_train_dos_probe, y_train)  # NBC is trained on the reduced train set and labels
    predicted = dos_probe_classifier.predict(X_test_dos_probe)  # Evaluates the model on the test data

    # Export the model as a pickle model
    if EXPORT_MODELS:
        try:
            file_name = 'dos_probe_NBC_classifier.pkl'
            with open(str('NSL-KDD Outputs/' + file_name), 'wb') as f:
                pickle.dump(dos_probe_classifier, f)
        except Exception as e:
            print(e)

    with open('NSL-KDD Outputs/Results.txt', 'a') as output_file:
        output_file.write('\n\n\nNSL-KDD Outputs for DoS+Probe Naive Bayesian Classifier:\n')
        output_file.write("\nConfusion Matrix: [TP FP / FN TN]\n" + str(confusion_matrix(y_test, predicted)))
        output_file.write('\nAccuracy = ' + str(accuracy_score(y_test, predicted)))
        output_file.write('\nF1 Score = ' + str(f1_score(y_test, predicted)))
        output_file.write('\nPrecision = ' + str(precision_score(y_test, predicted)))
        output_file.write('\nRecall = ' + str(recall_score(y_test, predicted)))
        output_file.write('\nShape of the train set: ' + str(X_train_dos_probe.shape))

    # Now that the DoS+Probe classifier has been built, focus on the r2l+u2r classifier
    df_train = copy.deepcopy(df_train_original)
    df_test = copy.deepcopy(df_test_original)

    # load from the train set only the targeted attacks categories (Normal+r2l+u2r)
    df_train = df_train[(df_train['label'] == 'normal') | (df_train['label'] == 'guess_passwd')
                        | (df_train['label'] == 'ftp_write')
                        | (df_train['label'] == 'imap') | (df_train['label'] == 'xsnoop') | (df_train['label'] == 'phf')
                        | (df_train['label'] == 'multihop') | (df_train['label'] == 'warezmaster')
                        | (df_train['label'] == 'xlock')
                        | (df_train['label'] == 'snmpguess') | (df_train['label'] == 'snmpgetattack')
                        | (df_train['label'] == 'httptunnel')
                        | (df_train['label'] == 'sendmail') | (df_train['label'] == 'named')
                        | (df_train['label'] == 'spy')
                        | (df_train['label'] == 'warezclient') | (df_train['label'] == 'buffer_overflow')
                        | (df_train['label'] == 'loadmodule') | (df_train['label'] == 'rootkit')
                        | (df_train['label'] == 'perl') | (df_train['label'] == 'xterm')
                        | (df_train['label'] == 'sqlattack')
                        | (df_train['label'] == 'ps')]

    # Sets the target label as 0 for normal traffic and 1 for u2r and r2l attacks
    y_train = np.array([0 if x == 'normal' else 1 for x in df_train['label']])
    df_train = df_train.drop(['label'], axis=1)
    df_train = df_train.reset_index().drop(['index'], axis=1)

    # Select the features from NSL-KDD to define u2r and r2l
    X_train = df_train[['srv_count', 'urgent', 'root_shell', 'hot',
                        'dst_host_srv_diff_host_rate', 'dst_host_count',
                        'logged_in', 'dst_host_same_src_port_rate', 'srv_diff_host_rate',
                        'num_shells', 'dst_host_srv_count']]

    scaler2 = MinMaxScaler()
    df_minmax = scaler2.fit_transform(X_train)  # Scales the features selected
    X_train = pd.DataFrame(df_minmax, columns=X_train.columns)

    # perform One-hot encoding
    ohe2 = OneHotEncoder(handle_unknown='ignore')
    label_enc = ohe2.fit_transform(df_train.iloc[:, 1:4])
    label_enc.toarray()
    new_labels = ohe2.get_feature_names_out(['protocol_type', 'service', 'flag'])
    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_train = pd.concat([X_train, df_enc], axis=1)

    # do the same for test set
    df_test = df_test[
        (df_test['label'] == 'normal') | (df_test['label'] == 'guess_passwd') | (df_test['label'] == 'ftp_write')
        | (df_test['label'] == 'imap') | (df_test['label'] == 'xsnoop') | (df_test['label'] == 'phf')
        | (df_test['label'] == 'multihop') | (df_test['label'] == 'warezmaster')
        | (df_test['label'] == 'xlock')
        | (df_test['label'] == 'snmpguess') | (df_test['label'] == 'snmpgetattack')
        | (df_test['label'] == 'httptunnel')
        | (df_test['label'] == 'sendmail') | (df_test['label'] == 'named')
        | (df_test['label'] == 'spy')
        | (df_test['label'] == 'warezclient') | (df_test['label'] == 'buffer_overflow')
        | (df_test['label'] == 'loadmodule') | (df_test['label'] == 'rootkit')
        | (df_test['label'] == 'perl') | (df_test['label'] == 'xterm')
        | (df_test['label'] == 'sqlattack')
        | (df_test['label'] == 'ps')]

    y_test = np.array([0 if x == 'normal' else 1 for x in df_test['label']])
    df_test = df_test.drop(['label'], axis=1)
    df_test = df_test.reset_index().drop(['index'], axis=1)

    # same for the test features
    X_test = df_test[['srv_count', 'urgent', 'root_shell', 'hot',
                      'dst_host_srv_diff_host_rate', 'dst_host_count',
                      'logged_in', 'dst_host_same_src_port_rate', 'srv_diff_host_rate',
                      'num_shells', 'dst_host_srv_count']]

    df_minmax = scaler2.transform(X_test)
    X_test = pd.DataFrame(df_minmax, columns=X_test.columns)

    label_enc = ohe2.transform(df_test.iloc[:, 1:4])
    label_enc.toarray()
    new_labels = ohe2.get_feature_names_out(['protocol_type', 'service', 'flag'])
    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_test = pd.concat([X_test, df_enc], axis=1)

    with open('NSL-KDD Outputs/Results.txt', 'a') as output_file:
        output_file.write("\n\n\nProperties of training set and test set for SVM:\n")
        output_file.write('\nShape of the training data features: (#samples, #features) =' + str(X_train.shape))
        output_file.write('\nShape of the training data labels: (#samples,) =' + str(y_train.shape))
        output_file.write('\nShape of the test data features: (#samples, #features) =' + str(X_test.shape))
        output_file.write('\nShape of the test data labels: (#samples,) =' + str(y_test.shape))

    # Performs under sampling, reduces the number of majority class samples to address class imbalance
    # instances of Normal traffic >>> U2R+R2L instances
    under_sampler = under_sam(sampling_strategy=1)
    # print(y_train)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    # PCA as done before on Dos+Probe
    pca_r2l_u2r = PCA(n_components=0.95)
    X_train_r2l_u2r = pca_r2l_u2r.fit_transform(X_train)
    X_test_r2l_u2r = pca_r2l_u2r.transform(X_test)

    # Support Vector Classifier with parameters C, gamma and kernel function 'radial basis function'
    r2l_u2r_classifier = SVC(C=0.1, gamma=0.01, kernel='rbf')
    r2l_u2r_classifier.fit(X_train_r2l_u2r, y_train)
    predicted = r2l_u2r_classifier.predict(X_test_r2l_u2r)

    # Export the datasets used for the NBC
    if EXPORT_DATASETS:
        X_train.to_csv('Updated Datasets/X_train_SVM.csv', index=False)
        y_train_df = pd.DataFrame({'label': y_train})
        y_train_df.to_csv('Updated Datasets/y_train_SVM.csv', index=False)

        X_test.to_csv('Updated Datasets/X_test_SVM.csv', index=False)
        y_test_df = pd.DataFrame({'label': y_test})
        y_test_df.to_csv('Updated Datasets/y_test_SVM.csv', index=False)

    # Export the model as a pickle model
    if EXPORT_MODELS:
        try:
            file_name = 'r2l_u2r_classifier.pkl'
            with open('NSL-KDD Outputs/' + file_name, 'wb') as f:
                pickle.dump(r2l_u2r_classifier, f)
        except Exception as e:
            print(e)

    with open('NSL-KDD Outputs/Results.txt', 'a') as output_file:
        output_file.write('\n\n\nNSL-KDD Outputs for u2r-r2l Support Vector Machine:\n')
        output_file.write("\nConfusion Matrix: [TP FP / FN TN]\n" + str(confusion_matrix(y_test, predicted)))
        output_file.write('\nAccuracy = ' + str(accuracy_score(y_test, predicted)))
        output_file.write('\nF1 Score = ' + str(f1_score(y_test, predicted)))
        output_file.write('\nPrecision = ' + str(precision_score(y_test, predicted)))
        output_file.write('\nRecall = ' + str(recall_score(y_test, predicted)))
        output_file.write('\nMatthew correlation coefficient = ' + str(matthews_corrcoef(y_test, predicted)))
        output_file.write('\nShape of the train set: ' + str(X_train_dos_probe.shape))

    # Testing
    df_test1 = copy.deepcopy(df_test_original)  # Create 2 copies of testing set, one for each layer
    df_test2 = copy.deepcopy(df_test_original)

    y_test_real = np.array([0 if x == 'normal' else 1 for x in df_test1['label']])  # 1=anomaly, 0=normal

    # Layer 1
    X_test = df_test1[['logged_in', 'count', 'serror_rate', 'rerror_rate',
                       'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                       'srv_diff_host_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate',
                       'dst_host_rerror_rate']]

    df_minmax = scaler1.transform(X_test)
    X_test = pd.DataFrame(df_minmax, columns=X_test.columns)
    label_enc = ohe.transform(df_test1.iloc[:, 1:4])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(['protocol_type', 'service', 'flag'])
    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_test = pd.concat([X_test, df_enc], axis=1)

    X_test_layer1 = pca_dos_probe.transform(X_test)

    # Layer 2
    X_test = df_test2[['srv_count', 'urgent', 'root_shell', 'hot',
                       'dst_host_srv_diff_host_rate', 'dst_host_count',
                       'logged_in', 'dst_host_same_src_port_rate', 'srv_diff_host_rate',
                       'num_shells', 'dst_host_srv_count']]

    df_minmax = scaler2.transform(X_test)
    X_test = pd.DataFrame(df_minmax, columns=X_test.columns)
    label_enc = ohe2.transform(df_test2.iloc[:, 1:4])
    label_enc.toarray()
    new_labels = ohe2.get_feature_names_out(['protocol_type', 'service', 'flag'])
    df_enc = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    X_test = pd.concat([X_test, df_enc], axis=1)

    X_test_layer2 = pca_r2l_u2r.transform(X_test)

    # Real testing
    classifier1 = dos_probe_classifier
    classifier2 = r2l_u2r_classifier

    result = []
    for i in range(22544):
        layer1 = classifier1.predict(X_test_layer1[i].reshape(1, -1))[0]
        if layer1 == 1:
            result.append(layer1)
        else:
            layer2 = classifier2.predict(X_test_layer2[i].reshape(1, -1))[0]
            if layer2 == 1:
                result.append(layer2)
            else:
                result.append(0)

    result = np.array(result)

    # the results may vary
    # C=0.1, gamma=0.01
    with open('NSL-KDD Outputs/Results.txt', 'a') as output_file:
        output_file.write('\n\n\nTesting the system:\n')
        output_file.write('\nShape of the test set for layer 1: ' + str(X_test_layer1.shape))
        output_file.write('\nShape of the test set for layer 2: ' + str(X_test_layer2.shape))
        output_file.write('\nConfusion Matrix: [TP FP / FN TN]\n' + str(confusion_matrix(y_test_real, result)))
        output_file.write('\nAccuracy = ' + str(accuracy_score(y_test_real, result)))
        output_file.write('\nF1 Score = ' + str(f1_score(y_test_real, result)))
        output_file.write('\nPrecision = ' + str(precision_score(y_test_real, result)))
        output_file.write('\nRecall = ' + str(recall_score(y_test_real, result)))
        output_file.write('\nMatthew corr = ' + str(matthews_corrcoef(y_test_real, result)))

    # Evaluate seen and unseen attack categories
    df_test = pd.read_csv('NSL-KDD_datasets/KDDTest+.txt', sep=",", header=None)
    df_test = df_test[df_test.columns[:-1]]
    df_test.columns = titles.to_list()
    y_test = df_test['label']
    df_test = df_test.drop(['num_outbound_cmds'], axis=1)
    df_test_original = df_test

    # List of unseen attacks
    new_attack = []
    for i in df_test_original['label'].value_counts().index.tolist()[1:]:
        if i not in df_train_original['label'].value_counts().index.tolist()[1:]:
            new_attack.append(i)

    new_attack.sort()

    # List of indexes of the new attacks compared to the original attack list
    index_of_new_attacks = []
    for i in range(len(df_test_original)):
        if df_test_original['label'][i] in new_attack:
            index_of_new_attacks.append(df_test_original.index[i])

    new_attack.append('normal')

    # List of indexes of the old attacks
    index_of_old_attacks = []
    for i in range(len(df_test_original)):
        if df_test_original['label'][i] not in new_attack:
            index_of_old_attacks.append(df_test_original.index[i])

    # Evaluate each attack type
    df_test = pd.read_csv('NSL-KDD_datasets/KDDTest+.txt', sep=",", header=None)
    df_test = df_test[df_test.columns[:-1]]
    df_test.columns = titles.to_list()
    y_test = df_test['label']
    df_test = df_test.drop(['num_outbound_cmds'], axis=1)
    df_test_original = df_test
    df = df_test_original

    dos_index = df.index[(df['label'] == 'apache2') | (df['label'] == 'back')
                         | (df['label'] == 'land') | (df['label'] == 'mailbomb')
                         | (df['label'] == 'neptune') | (df['label'] == 'pod')
                         | (df['label'] == 'processtable') | (df['label'] == 'smurf')
                         | (df['label'] == 'teardrop') | (df['label'] == 'udpstorm')
                         | (df['label'] == 'worm')].tolist()

    probe_index = df.index[(df['label'] == 'ipsweep') | (df['label'] == 'mscan')
                           | (df['label'] == 'nmap') | (df['label'] == 'portsweep')
                           | (df['label'] == 'saint') | (df['label'] == 'satan')].tolist()

    r2l_index = df.index[(df['label'] == 'ftp_write') | (df['label'] == 'guess_passwd')
                         | (df['label'] == 'httptunnel') | (df['label'] == 'imap')
                         | (df['label'] == 'multihop') | (df['label'] == 'named')
                         | (df['label'] == 'phf') | (df['label'] == 'sendmail')
                         | (df['label'] == 'snmpgetattack') | (df['label'] == 'snmpguess')
                         | (df['label'] == 'warezmaster') | (df['label'] == 'xlock')
                         | (df['label'] == 'xsnoop')].tolist()

    u2r_index = df.index[(df['label'] == 'buffer_overflow') | (df['label'] == 'loadmodule')
                         | (df['label'] == 'perl') | (df['label'] == 'ps')
                         | (df['label'] == 'rootkit') | (df['label'] == 'sqlattack')
                         | (df['label'] == 'xterm')].tolist()

    # Write the output on the text file
    with open('NSL-KDD Outputs/Results.txt', 'a') as output_file:
        output_file.write("\n\n\nTesting results:\n")

        output_file.write("\nTotal DoS samples: " + str(result[dos_index].shape[0]))
        output_file.write("\nDetected DoS: " + str(result[dos_index].sum()))
        output_file.write("\nDoS results correctness: " + str(result[dos_index].sum() / result[dos_index].shape[0]))

        output_file.write('\n\nTotal Probe samples: ' + str(result[probe_index].shape[0]))
        output_file.write('\nDetected Probe: ' + str(result[probe_index].sum()))
        output_file.write(
            '\nProbe results correctness: ' + (
                str(result[probe_index].sum() / result[probe_index].shape[0] * 100)) + '%')

        output_file.write('\n\nTotal R2L samples: ' + str(result[r2l_index].shape[0]))
        output_file.write('\nDetected R2L: ' + str(result[r2l_index].sum()))
        output_file.write(
            '\nR2L results correctness: ' + str(result[r2l_index].sum() / result[r2l_index].shape[0] * 100) + '%')

        output_file.write('\n\nTotal U2R samples: ' + str(result[u2r_index].shape[0]))
        output_file.write('\nDetected U2R: ' + str(result[u2r_index].sum()))
        output_file.write(
            '\nU2R results correctness: ' + (str(result[u2r_index].sum() / result[u2r_index].shape[0] * 100)) + '%')



main()
