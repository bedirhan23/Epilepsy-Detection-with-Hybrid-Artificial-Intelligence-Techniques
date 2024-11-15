import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from mrmr import mrmr_classif

def apply_pca(dataframe, n_components, verbose=False):
    dataframe = dataframe.drop(columns=['label'], axis=1)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(dataframe)
    principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    explained_variance_ratio = pca.explained_variance_ratio_

    if verbose:
        print(f"PCA Dataframe: {principalDf}")
        print(f"PCA Explained Variance Ratio: {explained_variance_ratio}")

    return principalDf


def apply_rfe(dataframe, target, n_features_to_select=5, estimator=None, verbose=False):
    dataframe = dataframe.drop(columns=['label'], axis=1)
    if estimator is None:
        raise ValueError("Please provide an estimator for RFE.")
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(dataframe, target)

    selected_features = dataframe.columns[selector.support_]
    selected_dataframe = dataframe[selected_features] 

    if verbose:
        print(f"RFE Dataframe: {dataframe}")
        print(f"Selected features: {selected_features}")

    return selected_dataframe


def apply_ica(dataframe, n_components, verbose=False):
    dataframe = dataframe.drop(columns=['label'], axis=1)
    ica = FastICA(n_components=n_components)
    transformed_features = ica.fit_transform(dataframe)
    transformed_df = pd.DataFrame(data=transformed_features, columns=[f'ICA_{i}' for i in range(1, n_components + 1)])
    explained_variance_ratio = np.sum(ica.components_ ** 2, axis=1) / np.sum(ica.components_ ** 2)

    if verbose:
        print(f"ICA dataframe: {transformed_df}")
        print(f"ICA explained Variance Ratio: {explained_variance_ratio}")

    return transformed_df


def apply_mrmr(dataframe, target, n_features_to_select, verbose=False):
    dataframe = dataframe.drop(columns=['label'], axis=1)
    selected_features = mrmr_classif(dataframe, target, n_features_to_select)
    selected_dataframe = dataframe[selected_features]

    if verbose:
        print(f"mRMR dataframe: {dataframe}")
        print(f"Selected features: {selected_features}")

    return selected_dataframe
