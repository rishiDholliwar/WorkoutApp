import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
import time

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes:.3f} \n'
    'kNN classifier:         {knn:.3f} \n'
    'Rand forest classifier: {rf:.3f} \n'
)

def main():
    data = pd.read_csv(sys.argv[1])
    print(data)
    X = data[['time', 'ax (m/s^2)', 'ay (m/s^2)','az (m/s^2)','aT (m/s^2)']].values
    y = data['Action'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    bayes_model = make_pipeline(
        #FunctionTransformer(convertToLABColour, validate=True),
        GaussianNB()
    )
    
    knn_model = make_pipeline(
        #FunctionTransformer(convertToLABColour, validate=True),
        KNeighborsClassifier(n_neighbors=5)
    )
    

    rf_model = make_pipeline(
        #FunctionTransformer(convertToHSVColour, validate=True),
        KNeighborsClassifier(n_neighbors=5)
    )    
    
    print("Training Started")
    # train each model 
    models = [bayes_model, knn_model, rf_model]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE.format(
        bayes=bayes_model.score(X_valid, y_valid),
        knn=knn_model.score(X_valid, y_valid),
        rf=rf_model.score(X_valid, y_valid),

    ))

    

if __name__ == '__main__':
    main()

