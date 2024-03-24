import os
import glob
import pathlib

import scipy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq
from scipy import integrate
from scipy import signal
import seaborn as sns
from tqdm.notebook import tqdm


COLUMNS = ['TACHO', 'UNDER_AXI', 'UNDER_TAN', 'UNDER_DIR', 'OVER_AXI', 'OVER_TAN', 'OVER_DIR', 'MICRO']
'''
column 1 tachometer signal that allows to estimate rotation frequency;
columns 2 to 4 underhang bearing accelerometer (axial, radiale tangential direction);
columns 5 to 7 overhang bearing accelerometer (axial, radiale tangential direction);
column 8 microphone.
'''
COLUMNS_NUM = 8
FS = 50_000
DURATION = 5
SAMPLES = FS * DURATION
# FFT_RES = DURATION


path_dir = r'D:\UserShare\MMFAB3\3.Project_data\9.Others\opendata\MAFAULDA\full'
files = glob.glob(path_dir + r'\**\*.csv', recursive=True)
FILES = len(files)

# progress = tqdm(total=len(files))

# for i,f in enumerate(files ):
#     path_ = f #os.path.join(r,f)
#     cat, lvl, rpm = get_cat_lvl(path_)

#     df_tmp = pd.read_csv(path_,header=None)

#     ary[i] = df_tmp.values

#     del df_tmp

#     progress.update(1)


# with open('ary.pkl','wb')as f:
#     pickle.dump(ary,f,protocol=4)
# with open('files.txt','wb')as f:
#     pickle.dump(files,f,protocol=4)


with open('ary.pkl', 'rb')as f:
    ary = pickle.load(f)
with open('files.txt', 'rb')as f:
    files = pickle.load(f)

# files *5
files_5x = []
[files_5x.extend([file] * 5) for file in files]

len(files_5x)

ary_5x = ary.reshape(1951 * 5, 250_000 // 5, 8)
ary_5x.shape

ary_5x_rfft = np.abs(rfft(ary_5x, axis=1))

ary_5x_rfft.shape

cats = [get_cat_lvl(file)[0] for file in files_5x]
lvls = [get_cat_lvl(file)[1] for file in files_5x]
rpms = [get_cat_lvl(file)[2] for file in files_5x]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_y = le.fit_transform(cats)
cat_classes = le.classes_

lvl_y = le.fit_transform(lvls)
lvl_classes = le.classes_

rpm_y = rpms

dataset = {'fft_data': ary_5x_rfft,

           'cat_y': cat_y,
           'cat_classes': cat_classes,

           'lvl_y': lvl_y,
           'lvl_classes': 'lvl_classes',

           'rpm_y': 'rpm_y'}

with open('dataset.pkl', 'wb')as f:
    pickle.dump(dataset, f, protocol=4)
