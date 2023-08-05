import os
from os.path import basename,dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

'''
fs = 50 kHz
duration = 5s
length = 250_000 s
'''
files = glob.glob(r'D:\dataset\MAFAULDA\imbalance\10g\*.csv')
file = files[0]
df_tmp = pd.read_csv(file,names=['tachometer',\
                                    'bearing1_axi','bearing1_rad','bearing1_tan',\
                                    'bearing2_axi','bearing2_rad','bearing2_tan',\
                                    'microphone'])

use_column = 'bearing1_rad'
data_dir = r'D:\dataset\MAFAULDA'

def get_condition(file):

    rpm = os.path.splitext(os.path.basename(file))[0]
    rpm = float(rpm)/65
    cat = basename(dirname(dirname(file)))
    lvl = basename(dirname(file))

    if cat =='MAFAULDA':
        cat = 'normal'
        lvl = 0
    elif cat =='imbalance':
        lvl = float(lvl.replace('g', ''))
        lvl = lvl /35
    elif cat == 'vertical-misalignment':
        lvl = float(lvl.replace('mm', ''))
        lvl = lvl / 1.9
    elif cat == 'horizontal-misalignment':
        lvl = float(lvl.replace('mm', ''))
        lvl = lvl / 2.0
    else:
        lvl = float(lvl.replace('g', ''))
        lvl = lvl /35

    return rpm,cat,lvl


def _load_data( file_path):
    df = pd.read_csv(file_path,names=['tachometer',\
                                      'bearing1_axi','bearing1_rad','bearing1_tan',\
                                      'bearing2_axi','bearing2_rad','bearing2_tan',\
                                      'microphone'])  # Assuming CSV format, adjust accordingly if using different file types
    data = df[use_column].values
    features = data
    conditions = get_condition(file_path)
    return features, conditions


all_features = []
all_conditions = []
# for file_path in file_paths:

split_length = 5_000
# assert len(features) % split_length==0

for r, ds, fs in os.walk(data_dir):
    fs = [f for f in fs if f.endswith('.csv')][:4]
    for f in fs:
        file_path = os.path.join(r, f)
        features, conditions = _load_data(file_path)

        condition_repeat_times = len(features)//split_length

        all_features.extend(features.reshape(-1,split_length))
        all_conditions.extend([conditions] * condition_repeat_times)

        # for i in range(len(features)//5_000):
        #
        #     all_features.append(features)
        #     all_conditions.append(conditions)






features = np.array(all_features)
conditions = np.array(all_conditions)
# features = np.concatenate(all_features, axis=0)
# conditions = np.concatenate(all_conditions, axis=0)