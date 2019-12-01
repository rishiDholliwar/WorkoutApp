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

def main():
    data = pd.read_csv(sys.argv[1])
    
    #used the following guide to test different models
    #https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    
    #data = data[data['Action']!= 'NONE']
    
    #print(data)
    #TODO: get more data, model may be overfiting
    #TODO: put in last n records as features as we are classifying motion not single records
    #X = data[['time', 'ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']].values
    X = data[['ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']].values
    y = data['Action'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        
    bayes_model = make_pipeline(
        #FunctionTransformer(convertToLABColour, validate=True),
        GaussianNB()
    )
    
    knn_model = make_pipeline(
        #FunctionTransformer(convertToLABColour, validate=True),
        KNeighborsClassifier(n_neighbors=10)
    )
    

    rf_model = make_pipeline(
        #FunctionTransformer(convertToHSVColour, validate=True),
        RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)
    )    
    
    gbc_model = make_pipeline(
        GradientBoostingClassifier(n_estimators=50, max_depth=4, min_samples_leaf=0.1)
    )  
 
    svc_model = make_pipeline(
        SVC(kernel='rbf', C=2.0,gamma='auto')
    )  
    
    
    # train each model 
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

    data['prediction'] = knn_model.predict(X);
    #print(data[['Action','prediction']])

    plt.ylabel("Exercise type")
    plt.xlabel("time")
    #plt.plot(data['time'], data['Action'], 'b.', alpha=0.5)
    plt.plot(data['time'], data['prediction'], 'b.', alpha=0.5)






if __name__ == '__main__':
    main()

