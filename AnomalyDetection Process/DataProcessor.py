import copy
import pandas as pd


def data_process(incoming_data, scaler, ohe, pca, features, cat_features):

    data = copy.deepcopy(incoming_data)
    to_scale = data[features]

    scaled = scaler.transform(to_scale)
    scaled_data = pd.DataFrame(scaled, columns=to_scale.columns)
    label_enc = ohe.transform(data[cat_features])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(cat_features)
    new_encoded = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    processed = pd.concat([scaled_data, new_encoded], axis=1)
    pca_transformed = pca.transform(processed)

    return pca_transformed
