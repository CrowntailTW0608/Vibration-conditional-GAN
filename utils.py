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
import os
import glob
import pathlib

from enum import Enum
import scipy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq
from scipy import integrate
from scipy import signal
import seaborn as sns
from tqdm import tqdm
from typing import List


COLUMNS = ['TACHO','UNDER_AXI','UNDER_RAD','UNDER_TAN','OVER_AXI','OVER_RAD','OVER_TAN','MICRO' ]

COLUMNS_NUM = 8
FS = 50_000
DURATION = 5
SAMPLES = FS * DURATION

use_columns = COLUMNS[1:3+1]
print(f'use columns : {use_columns}')

from typing import List


class FaultCat(Enum):
    NORMAL = 'normal'
    H_MIS = 'horizontal-misalignment'
    IMBALANCE = 'imbalance'
    V_MIS = 'vertical-misalignment'
    UNDER_BALL = r'underhang\ball_fault'
    UNDER_CAGE = r'underhang\cage_fault'
    UNDER_OUTER = r'underhang\outer_race'


def get_cat_lvl(path_):
    full_ind = path_.split('\\').index('full')
    l = path_.split('\\')[full_ind + 1:]

    rpm = float(l[-1].replace('.csv', ''))
    if len(l) == 2:
        cat = l[0]
        lvl = "0"
    elif len(l) == 3:
        cat = l[0]
        lvl = l[1]
    elif len(l) == 4:
        cat = f'{l[0]}_{l[1]}'
        lvl = l[2]

    if cat == 'normal':
        lvl = '0'

    elif cat == 'horizontal-misalignment':
        lvl = float(lvl.replace('mm', '')) / 2.0

    elif cat == 'imbalance':
        lvl = float(lvl.replace('g', '')) / 35

    elif cat == 'imbalance':
        lvl = float(lvl.replace('g', '')) / 35

    elif 'overhang' in cat or 'underhang' in cat:
        lvl = float(lvl.replace('g', '')) / 35

    elif cat == 'vertical-misalignment':
        lvl = float(lvl.replace('mm', '')) / 1.90

    else:
        raise

    lvl = round(float(lvl), 2)
    return cat, lvl, rpm


from typing import List


def mySplitTrainTest(l: list = []):
    trainList = []
    testList = []

    if len(l) != 0:
        trainList = l[::2]
        testList = l[1::2]
    return (trainList, testList)


def getCatPath(cat: FaultCat = FaultCat.NORMAL):
    basePath = r'D:\pyData\github\Vibration-conditional-GAN\data\full'

    if cat == FaultCat.NORMAL:
        path = os.path.join(basePath, fr'{FaultCat.NORMAL.value}\*.csv')

    else:
        path = os.path.join(basePath, rf'{cat.value}\*\*.csv')

    files = glob.glob(path, recursive=True)

    files = [f for f in files if 'datetime' not in f]

    return files





def getTraintest(llcat: List[FaultCat], train_test_split=True):
    ### get datafiles
    filesAll = []
    filesTrain = []
    filesTest = []

    for cat in llcat:
        files = getCatPath(cat)

        files = [f for f in files if int(get_cat_lvl(f)[2]) > 50]

        filesAll.extend(files)
        filesTrainSub, filesTestSub = mySplitTrainTest(files)

        filesTrain.extend(filesTrainSub)
        filesTest.extend(filesTestSub)

    lenTrain, lenTest, lenAll = len(filesTrain), len(filesTest), len(filesAll)

    def __get_data__(files: list):
        _n = 500
        _m = int(250_000 / _n)
        _len = len(files)
        _ary = np.zeros((_len, len(use_columns), 250_000))
        _ary50x = np.zeros((_len * _m, len(use_columns), _n))
        _labels = []
        _cats = []
        _rpms = []
        _lvls = []

        # proce train data
        with tqdm(total=_len, position=0, leave=True)as pbar:

            for i, f in enumerate(files):
                path_ = f  # os.path.join(r,f)
                cat, lvl, rpm = get_cat_lvl(path_)

                df_tmp = pd.read_csv(path_, names=COLUMNS)
                df_tmp = df_tmp[use_columns]
                _ary[i] = df_tmp.values.T

                del df_tmp
                pbar.update(1)

                # spice into 50 pcs
        for i in range(_m):
            _ary50x[_len * i:(i + 1) * _len, ...] = _ary[..., i * _n:(i + 1) * _n]

        # labels
        _cats = [get_cat_lvl(f)[0] for f in files] * _m
        _lvls = [get_cat_lvl(f)[1] for f in files] * _m
        _rpms = [get_cat_lvl(f)[2] for f in files] * _m

        return _ary50x, {'cats': _cats, 'levels': _lvls, 'rpms': _rpms}

    if train_test_split:

        aryTrain50x, labelTrain = __get_data__(filesTrain)
        aryTest50x, labelTest = __get_data__(filesTest)

    else:
        aryTrain50x, labelTrain = __get_data__(filesAll)
        aryTest50x, labelTest = np.array([]), {}


    print(f'train data: {aryTrain50x.shape} ,test data : {aryTest50x.shape}')
    return aryTrain50x, aryTest50x, labelTrain, labelTest



if __name__ =='__main__':

    useCats = [FaultCat.NORMAL,
               FaultCat.IMBALANCE]

    useCats = [FaultCat.NORMAL]

    aryTrain50x,aryTest50x,labelTrain,labelTest = getTraintest(useCats, train_test_split=False)

    obj = {'data': aryTrain50x,
           'label': labelTrain}

    with open(r'./data/ary_500x_NORMAL_rpmGT50.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)


