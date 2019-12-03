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
    #'Rand forest classifier: {rf:.3f} \n'
   # 'GradientBoostingClassifier: {gbc:.3f} \n'
    #'SVC: {svc:.3f} \n'
)

#used the following guide to test different models
#https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def main():
    combinedData = pd.read_csv("annotatedData/combinedData.csv")
    
    #clear data that is not in one of our classes
    #combinedData = combinedData[combinedData['Action']!= 'NONE']
    
    #50 samples is about 1/2 a second looking at the data
    PREV_RECORDS_WINDOW_LENGTH = 10
    
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
        GradientBoostingClassifier(n_estimators=200, max_depth=7, min_samples_leaf=1)
    )  
 
    svc_model = make_pipeline(
        #optional, knn has around the same score and SVC takes long to train
        StandardScaler(),
        SVC(kernel='linear', C=2.0,gamma='auto')
    )  
        
    # train models
    models = [bayes_model, knn_model]#, rf_model,gbc_model]#,svc_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        print(m.score(X_train, y_train))

    print(OUTPUT_TEMPLATE.format(
        bayes=bayes_model.score(X_valid, y_valid),
        knn=knn_model.score(X_valid, y_valid),
       # rf=rf_model.score(X_valid, y_valid),
        #gbc=gbc_model.score(X_valid,y_valid),
        #svc=svc_model.score(X_valid,y_valid)
    ))

    y_predicted = knn_model.predict(X_valid)
    
    from sklearn.metrics import classification_report
    print("KNeighborsClassifier Stats:")
    print(classification_report(y_valid, y_predicted))
    
    #y_predicted = gbc_model.predict(X_valid)
    #print("GradientBoostingClassifier Stats:")
    #print(classification_report(y_valid, y_predicted))


    mixedExercises1_df = pd.read_csv("annotatedData/annotated-mixedExercises1.csv")
    
    mixedDf =  mixedExercises1_df[['ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']]
    i2 = 1
    while i2 < PREV_RECORDS_WINDOW_LENGTH + 1:
        mixedDf['AxAtTminus'+str(i2)] = mixedDf['ax (m/s^2)'].shift(i)
        mixedDf['AyAtTminus'+str(i2)] = mixedDf['ay (m/s^2)'].shift(i)
        mixedDf['AzAtTminus'+str(i2)] = mixedDf['az (m/s^2)'].shift(i)
        mixedDf['AtAtTminus'+str(i2)] = mixedDf['aT (m/s^2)'].shift(i)
        i2+=1
    
    mixedDf = mixedDf.dropna();    
        
    mixed_pred = knn_model.predict(mixedDf)
    print("MixedExercise Prediction Stats:")
    print(classification_report(mixedExercises1_df['Action'][1:], mixed_pred))
   
    
    plt.subplot(2, 1, 1)
    plt.title("Predicted")
    plt.ylabel("exercise class")
    plt.xlabel("time")
    plt.plot(mixedExercises1_df['time'][1:],mixed_pred, 'b.', alpha=0.5)
    plt.savefig('mixedExercise-Predicted.png')
    
    #Used to format plots: https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle/45161551#45161551
    plt.figure().tight_layout(rect=[0, 0.03, 1, 0.95])
 
    plt.subplot(2, 1, 2)
    plt.title("Real")
    plt.ylabel("Real exercise class")
    plt.xlabel("time")
    plt.plot(mixedExercises1_df['time'][1:],mixedExercises1_df['Action'][1:], 'b.', alpha=0.5)
    plt.savefig('mixedExercise-Real.png')
    

if __name__ == '__main__':
    main()

