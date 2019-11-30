import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy.integrate as it
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import os


#Todo Remove time dont need

def create_exercise_data(inputFiles,exercises_df):
    for file in inputFiles: 
        df = pd.read_csv(file)
        print("Processing: ", file.name)
   
        t_max_avg = df.nlargest(5, 'aT (m/s^2)') # may want to take more values
        t_max_avg = t_max_avg['aT (m/s^2)'].mean()
        file = os.path.basename(file.name)
        file = ''.join([i for i in file if not i.isdigit()])
        action = file[:-4]
        exercises_df = exercises_df.append({'action': action,'t_max_avg':t_max_avg}
                                                            , ignore_index=True)

    return exercises_df

def main(inputFiles):
    exercises_df = pd.DataFrame(columns=['action', 't_max_avg'])
    exercises_df = create_exercise_data(inputFiles,exercises_df)
    print(exercises_df.groupby(['action']).agg([('average','mean')]),"\n")
    #print(exercises_df.groupby(['action']).agg([('median','median')]))
    print("Files Generated!")
    exercises_df.to_csv("exercise-model-data.csv", index=False)     

if __name__=='__main__':
    path = "model-data/"
    inputFiles = []
   
    for filename in os.listdir(path):
        print(filename)
        file  = path + filename
        inputFiles.append(open(file))
    main(inputFiles)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
