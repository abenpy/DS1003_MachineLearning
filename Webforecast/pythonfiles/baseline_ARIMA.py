
# coding: utf-8

# In[58]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[59]:

#Loadind the dataset 
traindf = pd.read_csv('downsample.csv')
traindf = traindf.fillna(0)
traindftrans = traindf.transpose()
traindftrans.drop('Page', inplace = True)
traindftrans.index = pd.bdate_range(start=traindftrans.index.values[0],
                                    periods=len(traindftrans), freq='D') #freq define timegaps



# parameter selection for ARIMA
ps = range(0, 3)
d = [1,2]
qs = range(0, 2)
from itertools import product

parameters = product(ps, d, qs)
parameters_list = list(parameters)

period = 180 #totally 803days, we predict last 60days


# Function for representing data in a linear form
def split_ap(values, period):
    l = list(values)
    split_list = lambda n: zip(*[iter(l + [None] * ((n - len(l) % n) % n))] * n)
    l = list(split_list(period))
    ap = []
    for i in l:
        ap.append(sum(i))
    temp = [0]
    for i in ap:
        b = temp[-1] + i
        temp.append(b)
    return temp[1:]

# SMAPE as Kaggle calculates it
def kaggle_smape(true, predicted):
    true_o = true
    pred_o = predicted
    summ = np.abs(true_o) + np.abs(pred_o)
    smape = np.where(summ==0, 0, 2*np.abs(pred_o - true_o) / summ)
    return smape


def findbestpara(df, parameters_list):
    best_smape = float("inf")
    best_paramlist = []
    best_mape_list = []
    for column in df: 
        dftrain = pd.DataFrame()
        dftrain['test'] = split_ap(df[column].values,1)
        dftrain.index = df.index
        for param in tqdm(parameters_list):
            # try except is needed, because on some sets of parameters the model is not trained
            try:
                model = sm.tsa.statespace.SARIMAX(dftrain.test[:563], order=(param[0], param[1], param[2]),
                                              seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
            
            except ValueError:
                continue
            except np.linalg.linalg.LinAlgError:
                continue

            forecast = model.forecast(steps=period)
            
            y_true, y_pred = dftrain.test.values[563:743], forecast[:]
            mape = []
            for i in range(180):
                mape_i = kaggle_smape(y_true[i],y_pred[i])
                mape.append(mape_i)
            ave_mape = np.mean(mape)
            # save the best model, aic, parameters
            if ave_mape < best_smape:
                best_model = model
                best_smape = ave_mape
                best_param = param
        best_mape_list.append(best_smape) 
        best_paramlist.append(best_param)
        best_smape = float("inf")
        warnings.filterwarnings('default')
        
    return best_paramlist, best_mape_list

best_paramlist,best_val_smape = findbestpara(traindftrans, parameters_list)


best_test_smape = []
for column in traindftrans:
    i = 0
    dftrain = pd.DataFrame()
    dftrain['test'] = split_ap(traindftrans[column].values,1)
    dftrain.index = traindftrans.index
    try:
        best_model = sm.tsa.statespace.SARIMAX(dftrain.test[:-60], order=(best_paramlist[i][0], best_paramlist[i][1], best_paramlist[i][2]),
                                      seasonal_order=(1, 1, 2, 12)).fit(disp=-1)
    except ValueError:
                continue
    forecast = best_model.get_prediction(start=pd.to_datetime(traindftrans.index.values[0]),
                                     end=pd.to_datetime(traindftrans.index.values[-1]), dynamic=False)
    forecast = forecast.predicted_mean
    predict = pd.DataFrame()
    predict['Real Value'] = traindftrans[column]
    # predict["Predict Value"] = forecast[-59:]
    predict['Real Value']=predict['Real Value'][-60:]
    predict["Predict Value"] = forecast - forecast.shift(1)
    predict["Predict Value"] = predict["Predict Value"][-60:]
    predict.dropna(inplace=True)

    valmape = []
    y_true = predict['Real Value']
    y_pred = predict["Predict Value"]
    for j in range(60):
        mape_i = kaggle_smape(y_true[j],y_pred[j])
        valmape.append(mape_i)
    ave_mape = np.mean(valmape)
    best_test_smape.append(ave_mape)
    i = i + 1
print(best_val_smape,best_test_smape)

np.mean(best_test_smape)

np.mean(best_val_smape)

