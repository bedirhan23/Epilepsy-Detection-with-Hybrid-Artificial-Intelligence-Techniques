import pandas as pd
from prepare_data import *
from feature_selection import *
#from logreg import *
from os.path import exists
from random_forest import rf_model
from sklearn.ensemble import RandomForestClassifier


def main():
    report()
    report(512)
    report(1024)
    report(level=4)
    report(512, level=4)
    report(1024, level=4)
    report(level=5)
    report(512, level=5)
    report(1024, level=5)
    report(wavelet='haar')
    report(512, 'haar')
    report(1024, 'haar')
    report(wavelet='haar', level=4)
    report(512, 'haar', level=4)
    report(1024, 'haar', level=4)
    report(wavelet='haar', level=5)
    report(512, 'haar', level=5)
    report(1024, 'haar', level=5)
    report(wavelet='sym6')
    report(512, 'sym6')
    report(1024, 'sym6')
    report(wavelet='sym6', level=4)
    report(512, 'sym6', level=4)
    report(1024, 'sym6', level=4)
    report(wavelet='sym6', level=5)
    report(512, 'sym6', level=5)
    report(1024, 'sym6', level=5)

def report(batch_size=256, wavelet='db5', level=3):
    df = all_cases(batch_size, wavelet, level)

    mrmr_metrics = []
    pca_metrics = []
    ica_metrics = []
    rfe_metrics = []

    best_params = {
        'bootstrap': False,
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 300
    }

    target = df['label']

    for i in range(5, 51, 5):
        mrmr_df = apply_mrmr(df, target, i)
        mrmr_df['label'] = df['label']

        pca_df = apply_pca(df, i)
        pca_df['label'] = df['label']

        ica_df = apply_ica(df, i)
        ica_df['label'] = df['label']

        rfe_df = apply_rfe(df, target, i, RandomForestClassifier(random_state=0))
        rfe_df['label'] = df['label']

        mrmr_metrics_list = rf_model(mrmr_df, best_params)
        pca_metrics_list = rf_model(pca_df, best_params)
        ica_metrics_list = rf_model(ica_df, best_params)
        rfe_metrics_list = rf_model(rfe_df, best_params)

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
    file_exists = exists('random_forest_report.xlsx')
    if file_exists:
        reader = pd.read_excel('random_forest_report.xlsx')
        with pd.ExcelWriter('random_forest_report.xlsx', mode='a', if_sheet_exists='overlay') as writer:
            mrmr_metrics_df.to_excel(writer, sheet_name='MRMR Metrics', index=False, header=False,
                                     startrow=len(reader) + 1)
            pca_metrics_df.to_excel(writer, sheet_name='PCA Metrics', index=False, header=False,
                                    startrow=len(reader) + 1)
            ica_metrics_df.to_excel(writer, sheet_name='ICA Metrics', index=False, header=False,
                                    startrow=len(reader) + 1)
            rfe_metrics_df.to_excel(writer, sheet_name='RFE Metrics', index=False, header=False,
                                    startrow=len(reader) + 1)
    else:
        with pd.ExcelWriter('random_forest_report.xlsx') as writer:
            mrmr_metrics_df.to_excel(writer, sheet_name='MRMR Metrics', index=False)
            pca_metrics_df.to_excel(writer, sheet_name='PCA Metrics', index=False)
            ica_metrics_df.to_excel(writer, sheet_name='ICA Metrics', index=False)
            rfe_metrics_df.to_excel(writer, sheet_name='RFE Metrics', index=False)

    return mrmr_metrics_df, pca_metrics_df, ica_metrics_df, rfe_metrics_df

if __name__ == '__main__':
    main()