import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def insertAction(df,start_time, end_time, action):
    df['Action'] = np.where(((df.time >= start_time) & 
                               (df.time <= end_time)), 
                                action, 
                                df['Action'])
    return df

def main():   
    mixedExercises1_df = pd.read_csv("rawData/mixedExercises1.csv")
    mixedExercises1_df['Action'] = 'NONE'
    
    #mixedExercises1_df = insertAction(exercises_df,0.0,6.0,"Putting phone in pocket")
    mixedExercises1_df = insertAction(mixedExercises1_df,6.0,12.0,"Walking")
    mixedExercises1_df = insertAction(mixedExercises1_df,12.0,31.0,"Squat")
    mixedExercises1_df = insertAction(mixedExercises1_df,31.0,37.0,"Stand")
    mixedExercises1_df = insertAction(mixedExercises1_df,37.0,49.0,"Walk")
    #mixedExercises1_df = insertAction(mixedExercises1_df,49.0,53.0,"WalkingToSitupTransition") #going down for situps
    mixedExercises1_df = insertAction(mixedExercises1_df,53.0,72.0,"Situp")
    #mixedExercises1_df = insertAction(mixedExercises1_df,72.0,75.0,"Up")
    #mixedExercises1_df = insertAction(mixedExercises1_df,75.0,78.0,"Down")
    mixedExercises1_df = insertAction(mixedExercises1_df,78.0,89.0,"Pushup")
    #mixedExercises1_df = insertAction(mixedExercises1_df,89.0,92.0,"Up")
    mixedExercises1_df = insertAction(mixedExercises1_df,92.0,99.0,"Walk")
    
    
    #process data 
    #calc deltaTime
    #mixedExercises1_df['nextTime'] = mixedExercises1_df['time'].shift(-1)
    #mixedExercises1_df = mixedExercises1_df.dropna();
    #mixedExercises1_df['deltaTime'] = mixedExercises1_df['nextTime'] - mixedExercises1_df['time']
    
    #chop out state transitions and junk data
    mixedExercises1_df = mixedExercises1_df[mixedExercises1_df['Action']!= 'NONE']
    #print(exercises_df)
    
    
    mixedExercises1_df.to_csv("annotatedData/annotatedData-1.csv",index=False)
    
if __name__=='__main__':
    #inputFile = sys.argv[1]
    #out_directory = sys.argv[2]
    main()