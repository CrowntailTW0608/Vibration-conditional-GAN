import os
from os.path import splitext, basename, dirname
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class VibrationDataset:

    def __init__(self, data_dir, use_columns:list, batch_size=32, shuffle=True, split_length=5_000, validation_split=0.1, test_split=0.1):

        self.data_dir = data_dir
        self.use_columns = use_columns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.test_split = test_split
        self.categories = sorted(os.listdir(self.data_dir))
        self.num_categories = len(self.categories)
        self.split_length = split_length
        self.N = 5 * 50_000

        assert self.N % split_length==0

    def _cat_2_num(self,cat):
        _dict = { 'normal':0,
                  'imbalance':1,
                  'horizontal-misalignment':2,
                  'vertical-misalignment':3,
                  'ball_fault':4,
                  'cage_fault':5,
                  'outer_race':6,}

        return _dict[cat]/6

    def _get_condition(self, file_path):

        rpm = os.path.splitext(basename(file_path))[0]
        rpm = float(rpm) / 65
        cat = basename(dirname(dirname(file_path)))
        lvl = basename(dirname(file_path))

        if cat == 'MAFAULDA':
            cat = 'normal'
            lvl = 0
        elif cat == 'imbalance':
            lvl = float(lvl.replace('g', ''))
            lvl = lvl / 35
        elif cat == 'vertical-misalignment':
            lvl = float(lvl.replace('mm', ''))
            lvl = lvl / 1.9
        elif cat == 'horizontal-misalignment':
            lvl = float(lvl.replace('mm', ''))
            lvl = lvl / 2.0
        else:
            lvl = float(lvl.replace('g', ''))
            lvl = lvl / 35
        cat = self._cat_2_num(cat)

        return rpm, cat, lvl

    def _load_data(self, file_path):
        df = pd.read_csv(file_path,names=['tachometer',\
                                          'bearing1_axi','bearing1_rad','bearing1_tan',\
                                          'bearing2_axi','bearing2_rad','bearing2_tan',\
                                          'microphone'])  # Assuming CSV format, adjust accordingly if using different file types
        data = df[self.use_columns].values
        features = data
        conditions = self._get_condition(file_path)
        return features, conditions

    def _load_all_data(self):

        all_features = []
        all_conditions = []

        for r, ds, fs in os.walk(self.data_dir):
            fs = [f for f in fs if f.endswith('.csv')]
            for f in fs:
                file_path = os.path.join(r, f)
                features, conditions = self._load_data(file_path)

                condition_repeat_times = self.N // self.split_length

                all_features.extend(features.reshape(-1, self.split_length))
                all_conditions.extend([conditions] * condition_repeat_times)
                # all_features.append(features)
                # all_conditions.append(conditions)

        features = np.array(all_features)
        conditions = np.array(all_conditions)

        # Perform feature scaling (optional, but can be beneficial for training)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, conditions

    def _preprocess_dataset(self, features, conditions):
        dataset = tf.data.Dataset.from_tensor_slices((features, conditions))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))

        dataset = dataset.batch(self.batch_size, drop_remainder=True) # 如果最後一個batch的數量小於batch_size，它會被丟棄，從而保證每個batch的大小都是相同的。
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def get_dataset(self):
        features, conditions = self._load_all_data()

        scaler = StandardScaler()
        features = scaler.fit_transform(features).reshape(-1, self.split_length, 1)

        dataset = self._preprocess_dataset(features, conditions)

        return dataset

    # def get_dataset_splits(self):
    #     features, conditions = self._load_all_data()
    #
    #     # 將整體數據集拆分成訓練集、驗證集和測試集
    #     x_train, x_test, y_train, y_test = train_test_split(features, conditions, test_size=self.test_split, random_state=42)
    #     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_split, random_state=42)
    #
    #     # 對特徵進行標準化
    #     scaler  = StandardScaler()
    #     x_train = scaler.fit_transform(x_train)
    #     x_val   = scaler.transform(x_val)
    #     x_test  = scaler.transform(x_test)
    #
    #     # 創建訓練集、驗證集和測試集的tf.data.Dataset
    #     train_dataset = self._preprocess_dataset(x_train, y_train)
    #     val_dataset   = self._preprocess_dataset(x_val, y_val)
    #     test_dataset  = self._preprocess_dataset(x_test, y_test)
    #
    #     return train_dataset, val_dataset, test_dataset



