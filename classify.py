import sys
import os
MyCustomModuleDirectory = os.path.abspath(r"D:\UserShare\MMFAB3\GaryGWLin\AIMS")
sys.path.append(MyCustomModuleDirectory)

import pandas as pd

from sklearn.metrics import confusion_matrix,f1_score, accuracy_score,ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from myUtils.myUtils import FaultCat,getTraintest

COLUMNS = ['TACHO','UNDER_AXI','UNDER_RAD','UNDER_TAN','OVER_AXI','OVER_RAD','OVER_TAN','MICRO' ]

COLUMNS_NUM = 8
FS = 50_000
DURATION = 5
SAMPLES = FS * DURATION


use_samples = 500
use_columns = COLUMNS[1:3+1]

print(f'use columns : {use_columns}')

df_reasult = pd.DataFrame(columns=['No','cats','rpmthreshold','use_samples','hidden size','max iter','test ratio','train f1','test f1'])

ind = 0
proce_dict = dict()
for useCats_ind in range(2,5): #3
    for rpm_threshold in [0,20,35,50]:

        useCats = [FaultCat.NORMAL, FaultCat.IMBALANCE,FaultCat.V_MIS,FaultCat.H_MIS]
        useCats = useCats[:useCats_ind]

        aryTrain50x,aryTest50x,labelTrain,labelTest = getTraintest(useCats, train_test_split=True,rpm_threshold=rpm_threshold)

        aryTrain50x,aryTest50x=aryTrain50x[:,:,:use_samples],aryTest50x[:,:,:use_samples]
        print(f'train data shape{aryTrain50x.shape}\ntest  data shape{aryTest50x.shape}')

        ## ,{'cats':_cats,'lavels':_lvls,'rpms':_rpms}
        catsTrain = labelTrain['cats']
        catsTest  = labelTest['cats']

        # lvlTrain = labelTrain['lavels']
        # lvlTest  = labelTest['lavels']

        rpmTrain = labelTrain['rpms']
        rpmTest  = labelTest['rpms']

        proce = make_pipeline(MinMaxScaler((0,1)),
                              MLPClassifier(hidden_layer_sizes=(300,),random_state=1, max_iter=300,verbose=False),
                              verbose=True)

        proce.fit(aryTrain50x.reshape(-1,use_samples * len(use_columns)),catsTrain)

        yTrainPred= proce.predict(aryTrain50x.reshape(-1,use_samples * len(use_columns)))
        train_f1 = f1_score(catsTrain,yTrainPred,average="macro")

        yTestPred = proce.predict(aryTest50x.reshape(-1,use_samples * len(use_columns)))
        test_f1 = f1_score(catsTest,yTestPred,average="macro")

        print(f'useCats : {useCats}  rpm_threshold:{rpm_threshold}  train f1 :{train_f1} test f1 :{test_f1}   ',)

        proce_dict[f"{rpm_threshold}_{'+'.join([ele.value for ele in useCats])}"] = proce

        df_reasult.loc[ind, 'No'] = ind
        df_reasult.loc[ind,'cats'] = '+'.join([ele.value for ele in useCats])
        df_reasult.loc[ind, 'rpmthreshold'] = rpm_threshold
        df_reasult.loc[ind, 'use_samples'] = 500
        df_reasult.loc[ind, 'hidden size'] = 300
        df_reasult.loc[ind, 'max iter'] = 300
        df_reasult.loc[ind, 'test ratio'] = 0.5
        df_reasult.loc[ind, 'train f1'] = train_f1
        df_reasult.loc[ind, 'test f1'] = test_f1

        ind+=1

'''
no,cats,rpmthreshold,hidden size,max iter,test ratio,train f1,test f1
0,normal+imblance,50,300,300,0.5,1.00,1.00
1,normal+imblance,30,300,300,0.5,0.98,0.94
2,normal+imblance,20,300,300,0.5,0.96,0.96
3,normal+imblance, 0,300,300,0.5,0.90,0.89

'''
df_reasult.to_csv(r'./Data/df_result.csv',index=False)



