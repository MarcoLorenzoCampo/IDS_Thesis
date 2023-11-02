import os

import DetectionSystem
import copy
import pandas as pd


def pipeline_data_process(ds: DetectionSystem, incoming_data, target_layer):
    """
    This function is used to process the incoming data:
    - The features are selected starting f

    :param ds: The detection system we need to do preprocessing for
    :param target_layer: Indicates if the data is processed to be fed to layer 1 or layer 2
    :param incoming_data: A single or multiple data samples to process, always in the format of a DataFrame
    :return: The processed data received as an input
    """

    data = copy.deepcopy(incoming_data)

    if target_layer == 1:
        to_scale = data[ds.kb.features_l1]
        scaler = ds.kb.scaler1
        ohe = ds.kb.ohe1
        pca = ds.kb.pca1
    else:
        to_scale = data[ds.kb.features_l2]
        scaler = ds.kb.scaler2
        ohe = ds.kb.ohe2
        pca = ds.kb.pca2

    scaled = scaler.transform(to_scale)
    scaled_data = pd.DataFrame(scaled, columns=to_scale.columns)
    label_enc = ohe.transform(data[ds.kb.cat_features])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(ds.kb.cat_features)
    new_encoded = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    processed = pd.concat([scaled_data, new_encoded], axis=1)
    pca_transformed = pca.transform(processed)

    return pca_transformed


class DataPreprocessingComponent:
    features = pd.read_csv('NSL-KDD Original Datasets/Field Names.csv', header=None)

    def traffic_quality_check(self, incoming_data):
        """
        This function checks if the incoming traffic has the correct features to be analyzed.
        Parameters:
        incoming_data (DataFrame): The incoming data to be checked.
        Returns:
        int: Returns 1 if all features match, otherwise returns 0.
        """

        for index, feature in enumerate(incoming_data.columns):
            if self.features[index] != feature:
                return 0

        return 1
