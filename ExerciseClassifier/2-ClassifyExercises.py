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
    #'SVC: {svc:.3f} \n'
)

#used the following guide to test different models
#https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def main():
    combinedData = pd.read_csv("annotatedData/combinedData.csv")
    
    #clear data that is not in one of our classes(Doing this decreases performance and training time)
    #combinedData = combinedData[combinedData['Action']!= 'NONE']
    
    #50 samples is about 1/2 a second looking at the data
    #set at 0 for disabled, increasing the value seems decrease performance
    #TODO: figrure out why this does not work
    PREV_RECORDS_WINDOW_LENGTH = 0
    
    #select data source for training
    data = combinedData
    
    Xdf = data[['ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']]
    
    i = 1
    while i < PREV_RECORDS_WINDOW_LENGTH + 1:
        Xdf['AxAtTminus'+str(i)] = Xdf['ax (m/s^2)'].shift(i)
        Xdf['AyAtTminus'+str(i)] = Xdf['ay (m/s^2)'].shift(i)
        Xdf['AzAtTminus'+str(i)] = Xdf['az (m/s^2)'].shift(i)
        Xdf['AtAtTminus'+str(i)] = Xdf['aT (m/s^2)'].shift(i)
        i+=1
        
    
    Xdf = Xdf.dropna();
    #print(Xdf)
    
    X = Xdf.values
    y = data.iloc[PREV_RECORDS_WINDOW_LENGTH:]['Action'].values
    #print(X)
    #print(y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        
    bayes_model = make_pipeline(
        StandardScaler(),    
        GaussianNB()
    )
    
    knn_model = make_pipeline(
        StandardScaler(),    
        KNeighborsClassifier(n_neighbors=2)
    )
    
    rf_model = make_pipeline(
        StandardScaler(),    
        RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)
    )    
    
    gbc_model = make_pipeline(
        StandardScaler(),    
        GradientBoostingClassifier(n_estimators=50, max_depth=4, min_samples_leaf=0.1)
    )  
 
    svc_model = make_pipeline(
        #optional, knn has around the same score and SVC takes long to train
        StandardScaler(),
        SVC(kernel='linear', C=2.0,gamma='auto')
    )  
        
    # train models
    models = [bayes_model, knn_model, rf_model,gbc_model]#,svc_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        print(m.score(X_train, y_train))

    print(OUTPUT_TEMPLATE.format(
        bayes=bayes_model.score(X_valid, y_valid),
        knn=knn_model.score(X_valid, y_valid),
        rf=rf_model.score(X_valid, y_valid),
        gbc=gbc_model.score(X_valid,y_valid),
        #svc=svc_model.score(X_valid,y_valid)
    ))

    y_predicted = knn_model.predict(X_valid)
    from sklearn.metrics import classification_report
    print("KNN_STATS")
    print(classification_report(y_valid, y_predicted))


if __name__ == '__main__':
    main()

