import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess

def insertAction(df,start_time, end_time, action):
    df['Action'] = np.where(((df.time >= start_time) & 
                               (df.time <= end_time)), 
                                action, 
                                df['Action'])
    return df

def main():   
    
    mixedExercises1_df = pd.read_csv("rawData/mixedExercises1.csv")
    #Annotate Data
    mixedExercises1_df['Action'] = 'NONE'
    mixedExercises1_df = insertAction(mixedExercises1_df,6.0,12.0,"Walk")
    mixedExercises1_df = insertAction(mixedExercises1_df,12.0,30.0,"Squat")
    mixedExercises1_df = insertAction(mixedExercises1_df,31.0,37.0,"Stand")
    mixedExercises1_df = insertAction(mixedExercises1_df,37.0,49.0,"Walk")
    mixedExercises1_df = insertAction(mixedExercises1_df,53.0,72.0,"Situp")
    mixedExercises1_df = insertAction(mixedExercises1_df,78.0,89.0,"Pushup")
    mixedExercises1_df = insertAction(mixedExercises1_df,92.0,99.0,"Walk")
    #Smooth data
    mixedExercises1_df['ax (m/s^2)'] = lowess(mixedExercises1_df['ax (m/s^2)'],mixedExercises1_df['time'], frac=0.1)
    mixedExercises1_df['ay (m/s^2)'] = lowess(mixedExercises1_df['ay (m/s^2)'],mixedExercises1_df['time'], frac=0.1)
    mixedExercises1_df['az (m/s^2)'] = lowess(mixedExercises1_df['az (m/s^2)'],mixedExercises1_df['time'], frac=0.1)
    mixedExercises1_df['aT (m/s^2)'] = lowess(mixedExercises1_df['aT (m/s^2)'],mixedExercises1_df['time'], frac=0.1) 
    #plot smoothed line
    #plt.ylabel("aT (m/s^2)")
    #plt.xlabel("time (s)")
    #plt.plot(mixedExercises1_df['time'], mixedExercises1_df['aT (m/s^2)'], 'b.', alpha=0.5)
    #loess_smoothed = lowess(mixedExercises1_df['aT (m/s^2)'],mixedExercises1_df['time'], frac=0.1)
    #plt.plot(mixedExercises1_df['time'], loess_smoothed[:, 1], 'r-')    

    
    
    pushup1_df = pd.read_csv("rawData/Pushup1.csv")
    #annotate
    pushup1_df['Action'] = 'NONE'       
    pushup1_df = insertAction(pushup1_df,10.0,30.0,"Pushup")
    #Smooth data
    pushup1_df['ax (m/s^2)'] = lowess(pushup1_df['ax (m/s^2)'],pushup1_df['time'], frac=0.1)
    pushup1_df['ay (m/s^2)'] = lowess(pushup1_df['ay (m/s^2)'],pushup1_df['time'], frac=0.1)
    pushup1_df['az (m/s^2)'] = lowess(pushup1_df['az (m/s^2)'],pushup1_df['time'], frac=0.1)
    pushup1_df['aT (m/s^2)'] = lowess(pushup1_df['aT (m/s^2)'],pushup1_df['time'], frac=0.1)     
    
    
    
    squat1_df = pd.read_csv("rawData/Squat1.csv")
    #annotate
    squat1_df['Action'] = 'NONE'       
    squat1_df = insertAction(pushup1_df,10.0,60.0,"Pushup")
    #Smooth data
    squat1_df['ax (m/s^2)'] = lowess(squat1_df['ax (m/s^2)'],squat1_df['time'], frac=0.1)
    squat1_df['ay (m/s^2)'] = lowess(squat1_df['ay (m/s^2)'],squat1_df['time'], frac=0.1)
    squat1_df['az (m/s^2)'] = lowess(squat1_df['az (m/s^2)'],squat1_df['time'], frac=0.1)
    squat1_df['aT (m/s^2)'] = lowess(squat1_df['aT (m/s^2)'],squat1_df['time'], frac=0.1)      
    
    
    
    #Export individual data
    mixedExercises1_df.to_csv("annotatedData/annotated-mixedExercises1.csv",index=False)
    pushup1_df.to_csv("annotatedData/annotated-pushup1.csv",index=False)
    squat1_df.to_csv("annotatedData/annotated-squat1.csv",index=False)
    
    #combine and export
    combinedData = mixedExercises1_df
    combinedData = combinedData.append(pushup1_df)
    combinedData = combinedData.append(squat1_df)
    combinedData.to_csv("annotatedData/combinedData.csv",index=False)
    
    
    
if __name__=='__main__':
    main()