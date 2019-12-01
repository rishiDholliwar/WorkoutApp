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

def main(inputFiles):   
    exercises_df = pd.read_csv(inputFiles)
    exercises_df['Action'] = 'NONE'
    
    #exercises_df = insertAction(exercises_df,0.0,6.0,"Putting phone in pocket")
    exercises_df = insertAction(exercises_df,6.0,12.0,"Walking")
    exercises_df = insertAction(exercises_df,12.0,31.0,"Squat")
    exercises_df = insertAction(exercises_df,31.0,37.0,"Stand")
    exercises_df = insertAction(exercises_df,37.0,49.0,"Walk")
    #exercises_df = insertAction(exercises_df,49.0,53.0,"WalkingToSitupTransition") #going down for situps
    exercises_df = insertAction(exercises_df,53.0,72.0,"Situp")
    #exercises_df = insertAction(exercises_df,72.0,75.0,"Up")
    #exercises_df = insertAction(exercises_df,75.0,78.0,"Down")
    exercises_df = insertAction(exercises_df,78.0,89.0,"Pushup")
    #exercises_df = insertAction(exercises_df,89.0,92.0,"Up")
    exercises_df = insertAction(exercises_df,92.0,99.0,"Walk")
    
    
    #exercises_df = exercises_df[exercises_df['Action']!= 'NA']
    #print(exercises_df)
    
    exercises_df.to_csv("annotatedData/annotatedData-1.csv",index=False)
    
if __name__=='__main__':
    inputFile = sys.argv[1]
    #out_directory = sys.argv[2]
    main(inputFile)