import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from os.path import basename, dirname
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import tqdm

names = ['tachometer',\
         'bearing1_axi','bearing1_rad','bearing1_tan',\
         'bearing2_axi','bearing2_rad','bearing2_tan',\
         'microphone']

class VibrationFeatureExtractor:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

        self.ftf_coe  = 0.3750 # CPM / rpm
        self.bpfo_coe = 2.9980 # CPM / rpm
        self.bpfi_coe = 5.0020 # CPM / rpm
        self.bsf_coe  = 1.8710  # CPM / rpm


    def time_domain_features(self, signal_df:pd.DataFrame):

        func_dict = {'mean':np.mean,
                     'std':np.std,
                     'rms': lambda x: np.sqrt(np.mean(x**2)),
                     'skewness':skew,
                     'kurt':kurtosis,
                     'crest_factor': lambda x:np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)))}


        dict_fea=dict()

        for col in signal_df.columns:
            values = signal_df[col]
            for func_name, func_ in func_dict.items():
                dict_fea['{}_{}'.format(col,func_name)] = func_(values)

        return dict_fea

    def _find_real_rps(self,values, rps:float):

        torlence_hz = 3  # hz
        N = len(values)
        resol = N/self.sampling_rate

        ind_from, ind_to = int(resol * (rps - torlence_hz)), int(resol * (rps + torlence_hz))

        fft_value = np.abs(np.fft.fft(values))[:N//2]
        fft_value = fft_value[ind_from: ind_to]

        fft_x = np.linspace(0, self.sampling_rate//2, N//2)
        fft_x = fft_x[ind_from: ind_to]

        fft_arg_ind = np.argmax(fft_value)
        real_rps = fft_x[fft_arg_ind]

        return real_rps


    def frequency_domain_features(self, signal_df:pd.DataFrame,rps:float):
        # 頻域特徵

        rps = self._find_real_rps(signal_df[signal_df.columns[0]].values,rps)

        torlence_hz = 2 #hz
        N = len(signal_df)

        dict_fea = dict()

        dict_freq = {
        'base_freq_05' : rps  * 0.5,
        'base_freq_1'  : rps  * 1,
        'base_freq_15' : rps  * 1.5,
        'base_freq_2'  : rps  * 2.0,
        'base_freq_25' : rps  * 0.5,
        'base_freq_3'  : rps  * 3,
        'base_freq_35' : rps  * 3.5,

        'ftf_freq_1' : rps  * self.ftf_coe,
        'ftf_freq_2' : rps  * self.ftf_coe * 2,

        'bpfo_freq_1' : rps  * self.bpfo_coe,
        'bpfo_freq_2' : rps  * self.bpfo_coe * 2,

        'bpfi_freq_1' : rps  * self.bpfi_coe,
        'bpfi_freq_2' : rps  * self.bpfi_coe * 2,

        'bsf_freq_1' : rps  * self.bsf_coe  ,
        'bsf_freq_2' : rps  * self.bsf_coe * 2,
        }
        resol = N / self.sampling_rate

        for col, fft_value in zip(signal_df.columns,np.abs(np.fft.fft(signal_df.values.T))):
            fft_value = fft_value[:N//2]

            for freq_name,freq in dict_freq.items():
                ind_from,ind_to = int(resol * (freq-torlence_hz)), int(resol * (freq+torlence_hz))
                dict_fea['{}_{}'.format(col,freq_name)] = fft_value[ ind_from:ind_to].max()


        return rps,dict_fea


    def extract_features(self, signal_df:pd.DataFrame, rps:float):

        # 提取時域特徵和頻域特徵
        dict_all_features = dict()

        time_domain_features = self.time_domain_features(signal_df)
        real_rps ,frequency_domain_features = self.frequency_domain_features(signal_df, rps)

        dict_all_features.update(time_domain_features)
        dict_all_features.update(frequency_domain_features)

        return real_rps,pd.DataFrame([dict_all_features])

class FeatureDtatset:
    def __init__(self, data_dir, split_length:int, use_column:str=None):

        self.data_dir=data_dir
        self.split_length = split_length
        if use_column:
            self.use_column = use_column
        else:
            self.use_column = ['bearing1_axi', 'bearing1_rad', 'bearing1_tan',
                               'bearing2_axi', 'bearing2_rad', 'bearing2_tan']
        self.my_feature_extractor = VibrationFeatureExtractor(50_000)

    def _get_label(self, file_path):

        rps = os.path.splitext(basename(file_path))[0]
        rps = float(rps)
        cat = basename(dirname(dirname(file_path)))
        lvl = basename(dirname(file_path))

        return (rps,cat,lvl)

    def _load_data(self, file_path):
        df = pd.read_csv(file_path, names=['tachometer', \
                                           'bearing1_axi', 'bearing1_rad', 'bearing1_tan', \
                                           'bearing2_axi', 'bearing2_rad', 'bearing2_tan', \
                                           'microphone'])  # Assuming CSV format, adjust accordingly if using different file types
        df = df[self.use_column]
        labels = self._get_label(file_path)
        return df, labels

    def _load_all_feature(self):

        all_features = pd.DataFrame()
        all_labels = []

        for r, ds, fs in os.walk(self.data_dir):
            fs = [f for f in fs if f.endswith('.csv')]
            print(r)
            for f in tqdm.tqdm(fs):
                file_path = os.path.join(r, f)
                data, labels = self._load_data(file_path)

                N = len(data)

                rps,cat,lvl = labels
                condition_repeat_times = N // self.split_length


                for i in range(condition_repeat_times):
                    df_slice = data[i*self.split_length:(i+1)*self.split_length]
                    real_rps,features = self.my_feature_extractor.extract_features(df_slice,rps=rps)

                    all_features = all_features.append(features)
                    all_labels.extend([labels])

        all_labels = np.array(all_labels)
        all_features = all_features.reset_index(drop=True)
        # Perform feature scaling (optional, but can be beneficial for training)
        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)

        return all_features, all_labels


if __name__ =='__main__':
    sampling_rate = 50_000  # 替換為你的採樣率
    feature_extractor = VibrationFeatureExtractor(sampling_rate)

    df_tmp = pd.read_csv(r'D:\dataset\MAFAULDA\overhang\ball_fault\35g\32.1536.csv',names=names)

    use_column = ['bearing1_axi', 'bearing1_rad', 'bearing1_tan',
                  'bearing2_axi', 'bearing2_rad', 'bearing2_tan']
    use_column = ['bearing1_rad']
    use_column = ['bearing1_axi']
    signal_df = df_tmp[use_column]


    real_rps, features = feature_extractor.extract_features(signal_df, rps=32.1536)

    print('real_rps',real_rps)

    my_fea = FeatureDtatset(r'D:\dataset\MAFAULDA',split_length=50_000)
    # features, labels = my_fea._load_all_feature()

    import pickle
    # features.to_csv(r'features.csv',index=False)
    # with open(r'labels.pkl', 'wb')as f:
    #     pickle.dump(labels, f)

    from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder,LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.utils import shuffle

    features = pd.read_csv(r'features.csv')
    # features = features[[col for col in features.columns if 'bearing1' in col and 'freq' not in col]]
    features = features[[col for col in features.columns if 'bearing1' in col ]]

    with open(r'labels.pkl', 'rb')as f:
        labels = pickle.load(f)
    labels_df = pd.DataFrame(labels, columns=['rps', 'cat', 'lvl'])

    rs = RobustScaler()
    X_norm = rs.fit_transform(features)

    le = LabelEncoder()
    y = labels[:, 1]
    y = le.fit_transform(y)

    train_condition = labels[:,0].astype(float)>25
    test_condition = labels[:,0].astype(float)<=25

    X_train,X_test = X_norm[train_condition],X_norm[test_condition]
    y_train,y_test = y[train_condition],y[test_condition]

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test,y_test = shuffle(X_test,y_test, random_state=42)

    # X_train,X_test,y_train,y_test = train_test_split(X_norm,y,train_size=0.8,random_state=42,shuffle=True,stratify=y)

    cls_rf = RandomForestClassifier()
    cls_rf.fit(X_train,y_train)

    y_pred = cls_rf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print(confusion_matrix(y_test,y_pred))
    print('acc',acc)

    feature_importance = cls_rf.feature_importances_

    print(feature_importance)

