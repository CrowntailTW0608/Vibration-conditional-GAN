import sys
import os
import random
import numpy as np
import pandas as pd
from scipy import fft
import matplotlib.pyplot as plt

MyCustomModuleDirectory = os.path.abspath(r"D:\UserShare\MMFAB3\GaryGWLin\AIMS")
sys.path.append(MyCustomModuleDirectory)

from sklearn.metrics import confusion_matrix,f1_score, accuracy_score,ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from myUtils.myUtils import FaultCat,getTraintest


import importlib
myDCGAN = importlib.import_module("DCGAN-1DCNN")



use_samples = 500

COLUMNS = ['TACHO','UNDER_AXI','UNDER_RAD','UNDER_TAN','OVER_AXI','OVER_RAD','OVER_TAN','MICRO' ]

COLUMNS_NUM = 8
FS = 50_000
DURATION = 5
SAMPLES = FS * DURATION


use_samples = 500
use_columns = COLUMNS[1:3+1]

print(f'use columns : {use_columns}')



cdcgan = myDCGAN.DCGAN()

# load model weights
cdcgan.load_discriminator(r'./exp/DCGAN-1D/imblance/DCGAN-1D_20240321081137/model/1000_discriminator_weights.ckpt')
cdcgan.load_generator(r'./exp/DCGAN-1D/imblance/DCGAN-1D_20240321081137/model/1000_generator_weights.ckpt')

gen_noise = np.random.normal(0, 1,  (20,100))
gen_imblances = cdcgan.generator(gen_noise).numpy()




cdcgan.load_discriminator(r'./exp/DCGAN-1D/normal/DCGAN-1D_20240320170334/model/1985_discriminator_weights.ckpt')
cdcgan.load_generator(r'./exp/DCGAN-1D/normal/DCGAN-1D_20240320170334/model/1985_generator_weights.ckpt')

gen_noise = np.random.normal(0, 1,  (20,100))
gen_normal = cdcgan.generator(gen_noise).numpy()






plt.plot(abs(fft(gen_imblances[0,1:,0]))[:250])
plt.show()



useCats = [FaultCat.NORMAL, FaultCat.IMBALANCE]
useCats = [FaultCat.NORMAL]

aryTrain50x, aryTest50x, labelTrain, labelTest = getTraintest(useCats, train_test_split=True,
                                                              rpm_threshold=50)

aryTrain50x, aryTest50x = aryTrain50x[:, :, :use_samples], aryTest50x[:, :, :use_samples]
print(f'train data shape{aryTrain50x.shape}\ntest  data shape{aryTest50x.shape}')


plt.plot(abs(fft(aryTrain50x[0,0,1:]))[:250])
plt.show()


raise

## ,{'cats':_cats,'lavels':_lvls,'rpms':_rpms}
catsTrain = labelTrain['cats']
catsTest = labelTest['cats']

# lvlTrain = labelTrain['lavels']
# lvlTest  = labelTest['lavels']

rpmTrain = labelTrain['rpms']
rpmeTst = labelTest['rpms']

# proce = make_pipeline(MinMaxScaler((0, 1)),
#                       MLPClassifier(hidden_layer_sizes=(300,), random_state=1, max_iter=300, verbose=False),
#                       verbose=True)

# proce.fit(aryTrain50x.reshape(-1, use_samples * len(use_columns)), catsTrain)
#
# y_gen_normal_pred = proce.predict(gen_normal.reshape(-1, use_samples * len(use_columns)))
#
# y_gen_imblance_pred = proce.predict(gen_imblances.reshape(-1, use_samples * len(use_columns)))
#
#
# print(y_gen_imblance_pred,'\n',y_gen_normal_pred)





