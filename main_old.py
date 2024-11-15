from prepare_data import *
from feature_selection import *
from sklearn.svm import SVR
# from logreg import *
# from dnn import *
from random_forest import *
import openpyxl
from svm import *


def main():
    # df = all_cases(batch_size=256, wavelet='db5', level=3)
    df = pd.read_csv("df_all_cases.csv")

    estimator = SVR(kernel="linear")
    target = df['label']

    pca_df = apply_pca(df, 5, verbose=False)
    ica_df = apply_ica(df, 5, verbose=False)
    rfe_df = apply_rfe(df, target, 5, estimator, verbose=False)
    mrmr_df = apply_mrmr(df, target, 5, verbose=False)

    mrmr_df['label'] = df['label']

    random_forest_model(mrmr_df)


def report(batch_size=256, wavelet='db5', level=3):
    # df = all_cases(batch_size=batch_size, wavelet=wavelet, level=level)
    df = pd.read_csv("df_all_cases.csv")

    mrmr_metrics = []
    pca_metrics = []
    ica_metrics = []
    rfe_metrics = []

    estimator = SVR(kernel="linear")
    target = df['label']

    for i in range(5, 30, 5):
        mrmr_df = apply_mrmr(df, target, i)
        mrmr_df['label'] = df['label']

        pca_df = apply_pca(df, i)
        pca_df['label'] = df['label']

        ica_df = apply_ica(df, i)
        ica_df['label'] = df['label']

        rfe_df = apply_rfe(df, target, i, estimator)
        rfe_df['label'] = df['label']

        mrmr_metrics_list = svm_model(mrmr_df)
        pca_metrics_list = svm_model(pca_df)
        ica_metrics_list = svm_model(ica_df)
        rfe_metrics_list = svm_model(rfe_df)

        mrmr_metrics.append(mrmr_metrics_list + [batch_size, wavelet, level, i])
        pca_metrics.append(pca_metrics_list + [batch_size, wavelet, level, i])
        ica_metrics.append(ica_metrics_list + [batch_size, wavelet, level, i])
        rfe_metrics.append(rfe_metrics_list + [batch_size, wavelet, level, i])

    mrmr_metrics_df = pd.DataFrame(mrmr_metrics,
                                   columns=['Accuracy', 'Error', 'ROC AUC', 'Precision', 'Recall', 'F1 Score',
                                            'Specificity', 'Window Size', 'Wavelet', 'Level', 'Feature Count'])
    pca_metrics_df = pd.DataFrame(pca_metrics,
                                  columns=['Accuracy', 'Error', 'ROC AUC', 'Precision', 'Recall', 'F1 Score',
                                           'Specificity', 'Window Size', 'Wavelet', 'Level', 'Feature Count'])
    ica_metrics_df = pd.DataFrame(ica_metrics,
                                  columns=['Accuracy', 'Error', 'ROC AUC', 'Precision', 'Recall', 'F1 Score',
                                           'Specificity', 'Window Size', 'Wavelet', 'Level', 'Feature Count'])
    rfe_metrics_df = pd.DataFrame(rfe_metrics,
                                  columns=['Accuracy', 'Error', 'ROC AUC', 'Precision', 'Recall', 'F1 Score',
                                           'Specificity', 'Window Size', 'Wavelet', 'Level', 'Feature Count'])

    with pd.ExcelWriter('svm_report.xlsx') as writer:
        mrmr_metrics_df.to_excel(writer, sheet_name='MRMR Metrics', index=False)
        pca_metrics_df.to_excel(writer, sheet_name='PCA Metrics', index=False)
        ica_metrics_df.to_excel(writer, sheet_name='ICA Metrics', index=False)
        rfe_metrics_df.to_excel(writer, sheet_name='RFE Metrics', index=False)

    return mrmr_metrics_df, pca_metrics_df, ica_metrics_df, rfe_metrics_df


report()

