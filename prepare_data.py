import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from sklearn.preprocessing import MinMaxScaler

def wavelet_transform(df, batch_size, wavelet, level):
    num_batches = len(df) // batch_size

    batches = [df[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    transformed_batches = []

    for batch in batches:
        data = np.array(batch).reshape(-1)

        coeffs = pywt.swt(data, wavelet, level=level)

        transformed_batches.append(coeffs)
    return transformed_batches

def visualize(transformed_batch):
    plt.figure(figsize=(10, 6))

    plt.subplot(len(transformed_batch)+1, 1, len(transformed_batch)-2)
    plt.ylabel('cA')
    plt.xlabel('Samples')
    plt.plot(np.arange(0, len(transformed_batch[0][0]), 1), transformed_batch[0][0].T)


    for i, coeffs_level in enumerate(transformed_batch):
        plt.subplot(len(transformed_batch)+1, 1, i+2)
        plt.ylabel(f'cD{3-i}')
        plt.xlabel('Samples')
        plt.plot(np.arange(0, len(coeffs_level[1]), 1), coeffs_level[1].T)

    plt.tight_layout()
    plt.show()

def entr(array):
    prob_dist = np.abs(array) / np.sum(np.abs(array))
    ent = entropy(prob_dist, base=2)
    return ent


def energy(array):
    return np.log(np.sum(np.abs(array) ** 2))


def zero_crossing_rate(array):
    return np.sum(np.abs(np.diff(np.sign(array))))


def power(array):
    return np.mean(np.abs(array) ** 2)


def mean_absolute_deviation(array):
    return np.mean(np.abs(array - np.mean(array)))


def median_absolute_deviation(array):
    return np.median(np.abs(array - np.median(array)))


def root_mean_square(array):
    return np.sqrt(np.mean(array ** 2))


def l1_norm(array):
    return np.linalg.norm(array, 1)


def l2_norm(array):
    return np.linalg.norm(array, 2)


def lmax_norm(array):
    return np.linalg.norm(array, np.inf)


def standard_deviation_of_slope(array):
    return np.std(np.diff(array))

def feature_extraction(batch, level):
    cA = batch[0][0]
    if level == 3:
        cD1 = batch[2][1]
        cD2 = batch[1][1]
        cD3 = batch[0][1]
    elif level == 4:
        cD1 = batch[3][1]
        cD2 = batch[2][1]
        cD3 = batch[1][1]
        cD4 = batch[0][1]
    elif level == 5:
        cD1 = batch[4][1]
        cD2 = batch[3][1]
        cD3 = batch[2][1]
        cD4 = batch[1][1]
        cD5 = batch[0][1]

    cA_features = [
        np.mean(cA),
        np.std(cA),
        np.median(cA),
        np.max(cA),
        np.min(cA),
        np.ptp(cA),
        kurtosis(cA),
        skew(cA),
        entr(cA),
        energy(cA),
        power(cA),
        zero_crossing_rate(cA),
        mean_absolute_deviation(cA),
        median_absolute_deviation(cA),
        root_mean_square(cA),
        l1_norm(cA),
        l2_norm(cA),
        lmax_norm(cA),
        standard_deviation_of_slope(cA),

    ]

    cD1_features = [
        np.mean(cD1),
        np.std(cD1),
        np.median(cD1),
        np.max(cD1),
        np.min(cD1),
        np.ptp(cD1),
        kurtosis(cD1),
        skew(cD1),
        entr(cD1),
        energy(cD1),
        power(cD1),
        zero_crossing_rate(cD1),
        mean_absolute_deviation(cD1),
        median_absolute_deviation(cD1),
        root_mean_square(cD1),
        l1_norm(cD1),
        l2_norm(cD1),
        lmax_norm(cD1),
        standard_deviation_of_slope(cD1),
    ]

    cD2_features = [
        np.mean(cD2),
        np.std(cD2),
        np.median(cD2),
        np.max(cD2),
        np.min(cD2),
        np.ptp(cD2),
        kurtosis(cD2),
        skew(cD2),
        entr(cD2),
        energy(cD2),
        power(cD2),
        zero_crossing_rate(cD2),
        mean_absolute_deviation(cD2),
        median_absolute_deviation(cD2),
        root_mean_square(cD2),
        l1_norm(cD2),
        l2_norm(cD2),
        lmax_norm(cD2),
        standard_deviation_of_slope(cD2),
    ]

    cD3_features = [
        np.mean(cD3),
        np.std(cD3),
        np.median(cD3),
        np.max(cD3),
        np.min(cD3),
        np.ptp(cD3),
        kurtosis(cD3),
        skew(cD3),
        entr(cD3),
        energy(cD3),
        power(cD3),
        zero_crossing_rate(cD3),
        mean_absolute_deviation(cD3),
        median_absolute_deviation(cD3),
        root_mean_square(cD3),
        l1_norm(cD3),
        l2_norm(cD3),
        lmax_norm(cD3),
        standard_deviation_of_slope(cD3),
    ]
    list_of_features = [cD1_features, cD2_features, cD3_features, cA_features]

    if level >= 4:
        cD4_features = [
            np.mean(cD4),
            np.std(cD4),
            np.median(cD4),
            np.max(cD4),
            np.min(cD4),
            np.ptp(cD4),
            kurtosis(cD4),
            skew(cD4),
            entr(cD4),
            energy(cD4),
            power(cD4),
            zero_crossing_rate(cD4),
            mean_absolute_deviation(cD4),
            median_absolute_deviation(cD4),
            root_mean_square(cD4),
            l1_norm(cD4),
            l2_norm(cD4),
            lmax_norm(cD4),
            standard_deviation_of_slope(cD4),
        ]
        list_of_features = [cD1_features, cD2_features, cD3_features, cD4_features, cA_features]

    if level == 5:
        cD5_features = [
            np.mean(cD5),
            np.std(cD5),
            np.median(cD5),
            np.max(cD5),
            np.min(cD5),
            np.ptp(cD5),
            kurtosis(cD5),
            skew(cD5),
            entr(cD5),
            energy(cD5),
            power(cD5),
            zero_crossing_rate(cD5),
            mean_absolute_deviation(cD5),
            median_absolute_deviation(cD5),
            root_mean_square(cD5),
            l1_norm(cD5),
            l2_norm(cD5),
            lmax_norm(cD5),
            standard_deviation_of_slope(cD5),
        ]
        list_of_features = [cD1_features, cD2_features, cD3_features, cD4_features, cD5_features, cA_features]

    return list_of_features


def create_df(case, batch_size, wavelet, level):
    dataframe = pd.DataFrame()
    columns = ['cD1', 'cD2', 'cD3', 'cA']
    if level == 4:
        columns = ['cD1', 'cD2', 'cD3', 'cD4', 'cA']
    if level == 5:
        columns = ['cD1', 'cD2', 'cD3', 'cD4', 'cD5', 'cA']
    stats = ['mean', 'std', 'med', 'max', 'min', 'range', 'kurt', 'skew', 'entropy', 'energy', 'power', 'zc', 'mad',
             'medianad', 'rms', 'l1', 'l2', 'lmax', 'sds']
    columns_list = []
    for y in range(level+1):
        column_prefix = columns[y]
        for stat in stats:
            columns_list.append(f"{column_prefix}_{stat}")

    dataframe = pd.DataFrame(columns=columns_list)


    for i in range(len(case)):
        df = case[i]
        transformed_batches = wavelet_transform(df, batch_size, wavelet, level)
        for j in range(len(transformed_batches)):
            if level == 3:
                cD1, cD2, cD3, cA = feature_extraction(transformed_batches[j], level)
                new_row = {f'cD1_mean': cD1[0], f'cD1_std': cD1[1], f'cD1_med': cD1[2],
                           f'cD1_max': cD1[3], f'cD1_min': cD1[4], f'cD1_range': cD1[5],
                           f'cD1_kurt': cD1[6], f'cD1_skew': cD1[7], f'cD1_entropy': cD1[8],
                           f'cD1_energy': cD1[9], f'cD1_power': cD1[10], f'cD1_zc': cD1[11],
                           f'cD1_mad': cD1[12], f'cD1_medianad': cD1[13], f'cD1_rms': cD1[14],
                           f'cD1_l1': cD1[15], f'cD1_l2': cD1[16], f'cD1_lmax': cD1[17], f'cD1_sds': cD1[18],
                           f'cD2_mean': cD2[0], f'cD2_std': cD2[1], f'cD2_med': cD2[2],
                           f'cD2_max': cD2[3], f'cD2_min': cD2[4], f'cD2_range': cD2[5],
                           f'cD2_kurt': cD2[6], f'cD2_skew': cD2[7], f'cD2_entropy': cD2[8],
                           f'cD2_energy': cD2[9], f'cD2_power': cD2[10], f'cD2_zc': cD2[11],
                           f'cD2_mad': cD2[12], f'cD2_medianad': cD2[13], f'cD2_rms': cD2[14],
                           f'cD2_l1': cD2[15], f'cD2_l2': cD2[16], f'cD2_lmax': cD2[17], f'cD2_sds': cD2[18],
                           f'cD3_mean': cD3[0], f'cD3_std': cD3[1], f'cD3_med': cD3[2],
                           f'cD3_max': cD3[3], f'cD3_min': cD3[4], f'cD3_range': cD3[5],
                           f'cD3_kurt': cD3[6], f'cD3_skew': cD3[7], f'cD3_entropy': cD3[8],
                           f'cD3_energy': cD3[9], f'cD3_power': cD3[10], f'cD3_zc': cD3[11],
                           f'cD3_mad': cD3[12], f'cD3_medianad': cD3[13], f'cD3_rms': cD3[14],
                           f'cD3_l1': cD3[15], f'cD3_l2': cD3[16], f'cD3_lmax': cD3[17], f'cD3_sds': cD3[18],
                           f'cA_mean': cA[0], f'cA_std': cA[1], f'cA_med': cA[2],
                           f'cA_max': cA[3], f'cA_min': cA[4], f'cA_range': cA[5],
                           f'cA_kurt': cA[6], f'cA_skew': cA[7], f'cA_entropy': cA[8],
                           f'cA_energy': cA[9], f'cA_power': cA[10], f'cA_zc': cA[11],
                           f'cA_mad': cA[12], f'cA_medianad': cA[13], f'cA_rms': cA[14],
                           f'cA_l1': cA[15], f'cA_l2': cA[16], f'cA_lmax': cA[17], f'cA_sds': cA[18]
                           }
            elif level == 4:
                cD1, cD2, cD3, cD4, cA = feature_extraction(transformed_batches[j], level)
                new_row = {f'cD1_mean': cD1[0], f'cD1_std': cD1[1], f'cD1_med': cD1[2],
                           f'cD1_max': cD1[3], f'cD1_min': cD1[4], f'cD1_range': cD1[5],
                           f'cD1_kurt': cD1[6], f'cD1_skew': cD1[7], f'cD1_entropy': cD1[8],
                           f'cD1_energy': cD1[9], f'cD1_power': cD1[10], f'cD1_zc': cD1[11],
                           f'cD1_mad': cD1[12], f'cD1_medianad': cD1[13], f'cD1_rms': cD1[14],
                           f'cD1_l1': cD1[15], f'cD1_l2': cD1[16], f'cD1_lmax': cD1[17], f'cD1_sds': cD1[18],
                           f'cD2_mean': cD2[0], f'cD2_std': cD2[1], f'cD2_med': cD2[2],
                           f'cD2_max': cD2[3], f'cD2_min': cD2[4], f'cD2_range': cD2[5],
                           f'cD2_kurt': cD2[6], f'cD2_skew': cD2[7], f'cD2_entropy': cD2[8],
                           f'cD2_energy': cD2[9], f'cD2_power': cD2[10], f'cD2_zc': cD2[11],
                           f'cD2_mad': cD2[12], f'cD2_medianad': cD2[13], f'cD2_rms': cD2[14],
                           f'cD2_l1': cD2[15], f'cD2_l2': cD2[16], f'cD2_lmax': cD2[17], f'cD2_sds': cD2[18],
                           f'cD3_mean': cD3[0], f'cD3_std': cD3[1], f'cD3_med': cD3[2],
                           f'cD3_max': cD3[3], f'cD3_min': cD3[4], f'cD3_range': cD3[5],
                           f'cD3_kurt': cD3[6], f'cD3_skew': cD3[7], f'cD3_entropy': cD3[8],
                           f'cD3_energy': cD3[9], f'cD3_power': cD3[10], f'cD3_zc': cD3[11],
                           f'cD3_mad': cD3[12], f'cD3_medianad': cD3[13], f'cD3_rms': cD3[14],
                           f'cD3_l1': cD3[15], f'cD3_l2': cD3[16], f'cD3_lmax': cD3[17], f'cD3_sds': cD3[18],
                           f'cD4_mean': cD4[0], f'cD4_std': cD4[1], f'cD4_med': cD4[2],
                           f'cD4_max': cD4[3], f'cD4_min': cD4[4], f'cD4_range': cD4[5],
                           f'cD4_kurt': cD4[6], f'cD4_skew': cD4[7], f'cD4_entropy': cD4[8],
                           f'cD4_energy': cD4[9], f'cD4_power': cD4[10], f'cD4_zc': cD4[11],
                           f'cD4_mad': cD4[12], f'cD4_medianad': cD4[13], f'cD4_rms': cD4[14],
                           f'cD4_l1': cD4[15], f'cD4_l2': cD4[16], f'cD4_lmax': cD4[17], f'cD4_sds': cD4[18],
                           f'cA_mean': cA[0], f'cA_std': cA[1], f'cA_med': cA[2],
                           f'cA_max': cA[3], f'cA_min': cA[4], f'cA_range': cA[5],
                           f'cA_kurt': cA[6], f'cA_skew': cA[7], f'cA_entropy': cA[8],
                           f'cA_energy': cA[9], f'cA_power': cA[10], f'cA_zc': cA[11],
                           f'cA_mad': cA[12], f'cA_medianad': cA[13], f'cA_rms': cA[14],
                           f'cA_l1': cA[15], f'cA_l2': cA[16], f'cA_lmax': cA[17], f'cA_sds': cA[18]
                           }


            elif level == 5:
                cD1, cD2, cD3, cD4, cD5, cA = feature_extraction(transformed_batches[j], level)
                new_row = {f'cD1_mean': cD1[0], f'cD1_std': cD1[1], f'cD1_med': cD1[2],
                           f'cD1_max': cD1[3], f'cD1_min': cD1[4], f'cD1_range': cD1[5],
                           f'cD1_kurt': cD1[6], f'cD1_skew': cD1[7], f'cD1_entropy': cD1[8],
                           f'cD1_energy': cD1[9], f'cD1_power': cD1[10], f'cD1_zc': cD1[11],
                           f'cD1_mad': cD1[12], f'cD1_medianad': cD1[13], f'cD1_rms': cD1[14],
                           f'cD1_l1': cD1[15], f'cD1_l2': cD1[16], f'cD1_lmax': cD1[17], f'cD1_sds': cD1[18],
                           f'cD2_mean': cD2[0], f'cD2_std': cD2[1], f'cD2_med': cD2[2],
                           f'cD2_max': cD2[3], f'cD2_min': cD2[4], f'cD2_range': cD2[5],
                           f'cD2_kurt': cD2[6], f'cD2_skew': cD2[7], f'cD2_entropy': cD2[8],
                           f'cD2_energy': cD2[9], f'cD2_power': cD2[10], f'cD2_zc': cD2[11],
                           f'cD2_mad': cD2[12], f'cD2_medianad': cD2[13], f'cD2_rms': cD2[14],
                           f'cD2_l1': cD2[15], f'cD2_l2': cD2[16], f'cD2_lmax': cD2[17], f'cD2_sds': cD2[18],
                           f'cD3_mean': cD3[0], f'cD3_std': cD3[1], f'cD3_med': cD3[2],
                           f'cD3_max': cD3[3], f'cD3_min': cD3[4], f'cD3_range': cD3[5],
                           f'cD3_kurt': cD3[6], f'cD3_skew': cD3[7], f'cD3_entropy': cD3[8],
                           f'cD3_energy': cD3[9], f'cD3_power': cD3[10], f'cD3_zc': cD3[11],
                           f'cD3_mad': cD3[12], f'cD3_medianad': cD3[13], f'cD3_rms': cD3[14],
                           f'cD3_l1': cD3[15], f'cD3_l2': cD3[16], f'cD3_lmax': cD3[17], f'cD3_sds': cD3[18],
                           f'cD4_mean': cD4[0], f'cD4_std': cD4[1], f'cD4_med': cD4[2],
                           f'cD4_max': cD4[3], f'cD4_min': cD4[4], f'cD4_range': cD4[5],
                           f'cD4_kurt': cD4[6], f'cD4_skew': cD4[7], f'cD4_entropy': cD4[8],
                           f'cD4_energy': cD4[9], f'cD4_power': cD4[10], f'cD4_zc': cD4[11],
                           f'cD4_mad': cD4[12], f'cD4_medianad': cD4[13], f'cD4_rms': cD4[14],
                           f'cD4_l1': cD4[15], f'cD4_l2': cD4[16], f'cD4_lmax': cD4[17], f'cD4_sds': cD4[18],
                           f'cD5_mean': cD5[0], f'cD5_std': cD5[1], f'cD5_med': cD5[2],
                           f'cD5_max': cD5[3], f'cD5_min': cD5[4], f'cD5_range': cD5[5],
                           f'cD5_kurt': cD5[6], f'cD5_skew': cD5[7], f'cD5_entropy': cD5[8],
                           f'cD5_energy': cD5[9], f'cD5_power': cD5[10], f'cD5_zc': cD5[11],
                           f'cD5_mad': cD5[12], f'cD5_medianad': cD5[13], f'cD5_rms': cD5[14],
                           f'cD5_l1': cD5[15], f'cD5_l2': cD5[16], f'cD5_lmax': cD5[17], f'cD5_sds': cD5[18],
                           f'cA_mean': cA[0], f'cA_std': cA[1], f'cA_med': cA[2],
                           f'cA_max': cA[3], f'cA_min': cA[4], f'cA_range': cA[5],
                           f'cA_kurt': cA[6], f'cA_skew': cA[7], f'cA_entropy': cA[8],
                           f'cA_energy': cA[9], f'cA_power': cA[10], f'cA_zc': cA[11],
                           f'cA_mad': cA[12], f'cA_medianad': cA[13], f'cA_rms': cA[14],
                           f'cA_l1': cA[15], f'cA_l2': cA[16], f'cA_lmax': cA[17], f'cA_sds': cA[18]
                           }

            dataframe.loc[len(dataframe)] = new_row
    return dataframe

def read_file(directory):
    temp_list = []
    for file in os.listdir(directory):
        fl = directory + file
        temp_list.append(fl)
    temp_list = sorted(temp_list)
    all_files = []
    st = 'A'
    for i in range(len(temp_list)):
        x = pd.read_csv(temp_list[i],header=None)
        x.columns=[st+str(i+1)]
        all_files.append(x)
    return all_files

def normalize(df):
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_norm


def all_cases(batch_size=256, wavelet='db5', level=3):
    caseZ = read_file('./Z/')
    caseS = read_file('./S/')
    caseO = read_file('./O/')
    caseN = read_file('./N/')
    caseF = read_file('./F/')

    dfZ = create_df(caseZ, batch_size, wavelet, level)
    dfS = create_df(caseS, batch_size, wavelet, level)
    dfO = create_df(caseO, batch_size, wavelet, level)
    dfN = create_df(caseN, batch_size, wavelet, level)
    dfF = create_df(caseF, batch_size, wavelet, level)

    dfZ = normalize(dfZ)
    dfS = normalize(dfS)
    dfO = normalize(dfO)
    dfN = normalize(dfN)
    dfF = normalize(dfF)

    dfZ["Z"] = 1
    dfS["S"] = 1
    dfO["O"] = 1
    dfN["N"] = 1
    dfF["F"] = 1

    data = pd.concat([dfZ, dfS, dfO, dfN, dfF], ignore_index=True)
    data = data.fillna(0)

    data["label"] = data.apply(lambda row: 0 if row["Z"] == 1 else (1 if row["S"] == 1 else (2 if row["O"] == 1 else (3 if row["N"] == 1 else 4))), axis=1)

    data = data.drop(columns=['Z', 'S', 'O', 'N', 'F'])

    return data


if __name__ == '__main__':
    df_all_cases = all_cases()
    df_all_cases.to_csv('df_all_cases.csv', index=False)
    print(df_all_cases)


"""final_df = all_cases(512)
final_df = all_cases(1024)
final_df = all_cases(level=4)
final_df = all_cases(512, level=4)
final_df = call_cases(1024, level=4)
final_df = all_cases(level=5)
final_df = all_cases(512, level=5)
final_df = all_cases(1024, level=5)
final_df = all_cases(wavelet='haar')
final_df = all_cases(512, 'haar')
final_df = all_cases(1024, 'haar')
final_df = all_cases(wavelet='haar', level=4)
final_df = all_cases(512, 'haar', level=4)
final_df = all_cases(1024, 'haar', level=4)
final_df = all_cases(wavelet='haar', level=5)
final_df = all_cases(512, 'haar', level=5)
final_df = all_cases(1024, 'haar', level=5)
final_df = all_cases(wavelet='sym4')
final_df = all_cases(512, 'sym4')
final_df = all_cases(1024, 'sym4')
final_df = all_cases(wavelet='sym4', level=4)
final_df = all_cases(512, 'sym4', level=4)
final_df = all_cases(1024, 'sym4', level=4)
final_df = all_cases(wavelet='sym4', level=5)
final_df = all_cases(512, 'sym4', level=5)
final_df = all_cases(1024, 'sym4', level=5)
final_df = all_cases(wavelet='sym6')
final_df = all_cases(512, 'sym6')
final_df = all_cases(1024, 'sym6')
final_df = all_cases(wavelet='sym6', level=4)
final_df = all_cases(512, 'sym6', level=4)
final_df = all_cases(1024, 'sym6', level=4)
final_df = all_cases(wavelet='sym6', level=5)
final_df = all_cases(512, 'sym6', level=5)
final_df = all_cases(1024, 'sym6', level=5)
final_df = all_cases(wavelet='sym9')
final_df = all_cases(512, 'sym9')
final_df = all_cases(1024, 'sym9')
final_df = all_cases(wavelet='sym9', level=4)
final_df = all_cases(512, 'sym9', level=4)
final_df = all_cases(1024, 'sym9', level=4)
final_df = all_cases(wavelet='sym9', level=5)
final_df = all_cases(512, 'sym9', level=5)
final_df = all_cases(1024, 'sym9', level=5)
final_df = all_cases(wavelet='sym20')
final_df = all_cases(512, 'sym20')
final_df = all_cases(1024, 'sym20')
final_df = all_cases(wavelet='sym20', level=4)
final_df = all_cases(512, 'sym20', level=4)
final_df = all_cases(1024, 'sym20', level=4)
final_df = all_cases(wavelet='sym20', level=5)
final_df = all_cases(512, 'sym20', level=5)
final_df = all_cases(1024, 'sym20', level=5)
final_df = all_cases(wavelet='coif5')
final_df = all_cases(512, 'coif5')
final_df = all_cases(1024, 'coif5')
final_df = all_cases(wavelet='coif5', level=4)
final_df = all_cases(512, 'coif5', level=4)
final_df = all_cases(1024, 'coif5', level=4)
final_df = all_cases(wavelet='coif5', level=5)
final_df = all_cases(512, 'coif5', level=5)
final_df = all_cases(1024, 'coif5', level=5)
final_df = all_cases(wavelet='coif15')
final_df = all_cases(512, 'coif15')
final_df = all_cases(1024, 'coif15')
final_df = all_cases(wavelet='coif15', level=4)
final_df = all_cases(512, 'coif15', level=4)
final_df = all_cases(1024, 'coif15', level=4)
final_df = all_cases(wavelet='coif15', level=5)
final_df = all_cases(512, 'coif15', level=5)
final_df = all_cases(1024, 'coif15', level=5)
final_df = all_cases(wavelet='bior6.8')
final_df = all_cases(512, 'bior6.8')
final_df = all_cases(1024, 'bior6.8')
final_df = all_cases(wavelet='bior6.8', level=4)
final_df = all_cases(512, 'bior6.8', level=4)
final_df = all_cases(1024, 'bior6.8', level=4)
final_df = all_cases(wavelet='bior6.8', level=5)
final_df = all_cases(512, 'bior6.8', level=5)
final_df = all_cases(1024, 'bior6.8', level=5)
final_df = all_cases(wavelet='rbio6.8')
final_df = all_cases(512, 'rbio6.8')
final_df = all_cases(1024, 'rbio6.8')
final_df = all_cases(wavelet='rbio6.8', level=4)
final_df = all_cases(512, 'rbio6.8', level=4)
final_df = all_cases(1024, 'rbio6.8', level=4)
final_df = all_cases(wavelet='rbio6.8', level=5)
final_df = all_cases(512, 'rbio6.8', level=5)
final_df = all_cases(1024, 'rbio6.8', level=5)
final_df = all_cases(wavelet='dmey')
final_df = all_cases(512, 'dmey')
final_df = all_cases(1024, 'dmey')
final_df = all_cases(wavelet='dmey', level=4)
final_df = all_cases(512, 'dmey', level=4)
final_df = all_cases(1024, 'dmey', level=4)
final_df = all_cases(wavelet='dmey', level=5)
final_df = all_cases(512, 'dmey', level=5)
final_df = all_cases(1024, 'dmey', level=5)"""






                





