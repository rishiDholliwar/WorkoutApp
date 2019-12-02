import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
import time

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes:.3f} \n'
    'kNN classifier:         {knn:.3f} \n'
    'Rand forest classifier: {rf:.3f} \n'
    'GradientBoostingClassifier: {gbc:.3f} \n'
    'SVC: {svc:.3f} \n'
)

#used the following guide to test different models
#https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def main():
    combinedData = pd.read_csv("annotatedData/combinedData.csv")
    
    #clear data that is not in one of our classes(Is this a good idea?)
    #combinedData = combinedData[combinedData['Action']!= 'NONE']
    
    #history window to look back at
    #50 samples is about 1/2 a second looking at the data
    LOOKBACK_WINDOW_LENGTH = 2
    
 
    #select data source for training
    data = combinedData
    
        
    #TODO: get more data, model may be overfitting
    #TODO: put in last n records as features as we are classifying motion not single records
    #TODO: make history data be based off of LOOKBACK_WINDOW_LENGTH 
    Xdf = data[['ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']]
    Xdf['lastAx'] = Xdf['ax (m/s^2)'].shift(1)
    Xdf['lastAy'] = Xdf['ay (m/s^2)'].shift(1)
    Xdf['lastAz'] = Xdf['az (m/s^2)'].shift(1)
    Xdf['lastAt'] = Xdf['aT (m/s^2)'].shift(1)
    Xdf['AxAtTminus2'] = Xdf['ax (m/s^2)'].shift(2)
    Xdf['AyAtTminus2'] = Xdf['ay (m/s^2)'].shift(2)
    Xdf['AzAtTminus2'] = Xdf['az (m/s^2)'].shift(2)
    Xdf['AtAtTminus2'] = Xdf['aT (m/s^2)'].shift(2)
    Xdf = Xdf.dropna();
    #print(Xdf)
    #print(data)
    #print(data.iloc[1:])
    
    X = Xdf.values
    y = data.iloc[LOOKBACK_WINDOW_LENGTH:]['Action'].values
    #print(X)
    #print(y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        
    bayes_model = make_pipeline(
        GaussianNB()
    )
    
    knn_model = make_pipeline(
        KNeighborsClassifier(n_neighbors=10)
    )
    

    rf_model = make_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)
    )    
    
    gbc_model = make_pipeline(
        GradientBoostingClassifier(n_estimators=50, max_depth=4, min_samples_leaf=0.1)
    )  
 
    svc_model = make_pipeline(
        #optional, knn has around the same score and SVC takes long to train
        SVC(kernel='rbf', C=2.0,gamma='auto')
    )  
    
    
    # train models
    models = [bayes_model, knn_model, rf_model,gbc_model,svc_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        print(m.score(X_train, y_train))

    print(OUTPUT_TEMPLATE.format(
        bayes=bayes_model.score(X_valid, y_valid),
        knn=knn_model.score(X_valid, y_valid),
        rf=rf_model.score(X_valid, y_valid),
        gbc=gbc_model.score(X_valid,y_valid),
        svc=svc_model.score(X_valid,y_valid)

    ))

    #prediction = knn_model.predict(X);


if __name__ == '__main__':
    main()

