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

"""
    TODO fix standing 
    TODO add more actions like climbing stairs ...     
"""

def get_workout_actions(seconds,df,model):
    workout_actions = []
    elapsed_time = df["time"].iloc[-1] 
    low = 1
    high = seconds
    
    while high < elapsed_time:
        subset_df =  df[((df.time >=low) & (df.time < high))]
        #print("subset for < ", high)
        #print(subset_df.head(1).time)
        #print(subset_df.tail(1).time)
        t_max_avg = subset_df.nlargest(5, 'aT (m/s^2)') # may want to take more values
        t_max_avg = t_max_avg['aT (m/s^2)'].mean()
        #print("t_max_avg=",t_max_avg)
        prediction = get_action(subset_df,model)
        #print("Action predicted is: ", prediction,"\n")
        low = high
        high += seconds
        workout_actions.append(prediction)
        
    return  workout_actions

def print_timeline(workout_actions,time_spent_action,elapsed_time):
    prev_action = workout_actions[0]
    prev_action_time = 0
    current_action_time = 0
    del workout_actions[0]
    print("Workout Timeline:")
    
    for action in workout_actions:
        if(prev_action == action):
            current_action_time += time_spent_action
        else:
            print("- Time ["+ str(prev_action_time) + "s-", str(current_action_time) + "s]", " = " + prev_action)
            prev_action_time = current_action_time
            prev_action = action
            
    print("- Time [" + str(prev_action_time) + "s-", str(elapsed_time) +"s]", " = " + action)
    
    
    
def get_action(df,model):
    t_max_avg = df.nlargest(5, 'aT (m/s^2)') # may want to take more values
    t_max_avg = t_max_avg['aT (m/s^2)'].mean()
    
    #predict_df = pd.DataFrame(columns=['x_avg', 'y_avg', 'z_avg'])
    #predict_df = predict_df.append({'x_avg': x_avg, 'y_avg': y_avg, 'z_avg': z_avg   }, ignore_index=True)
    predict_df = pd.DataFrame(columns=['t_max_avg'])
    predict_df = predict_df.append({'t_max_avg': t_max_avg }, ignore_index=True)

    return model.predict(predict_df)[0]

#calories burned formula from:
#https://www.hss.edu/conditions_burning-calories-with-exercise-calculating-estimated-energy-expenditure.asp
def get_calories_burned(time_spent_action, action, weight):
    if action == 'walk':
        mets = 3.5
    elif action == 'jog':
        mets = 8
    elif action == 'run':
        mets = 12
    else:
        mets = 1.2 #Standing
    
    minutes = time_spent_action/60
    weight = 90 # in kg 90kg = 200 pounds 
    calories_burned = (.0175 * mets * weight) * minutes
    return round(calories_burned,2)
    
    
def print_workout_stats(workout_actions,elapsed_time):
    workout = Counter(workout_actions)
    numActions = len(workout_actions)
    weight = 90
    total_cal_burned = 0
    workout_breakdown = ""
   
    for k,v in workout.items():
        percentage = str(round((v/numActions)*100,2))
        time_spent_action = (v/numActions)*elapsed_time
        calories_burned = get_calories_burned(time_spent_action,k,weight)
        total_cal_burned += calories_burned
        workout_breakdown += (" - " + k +": " +  percentage + "%" + ", " 
                              + str(calories_burned) + " calories" +"\n")
        
    print("Workout Overview:")
    workout_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    print(" - Time: " , workout_time + "\n - Calories Burned: ", total_cal_burned,"\n")
    print("Workout Breakdown:")
    print(workout_breakdown)
    plt.title("Workout Breakdown")
    plt.pie(workout.values(),  labels=workout.keys(), autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #plt.show()
    
def main(exercise_labelled,workout):
    exercise_labelled_df = pd.read_csv(exercise_labelled)
    workout_df = pd.read_csv(workout)
    
    #plt.plot(guess_exercise_df['time'], guess_exercise_df['aT (m/s^2)'], 'b.', alpha=0.5)
    #loess_smoothed = lowess(guess_exercise_df['aT (m/s^2)'],guess_exercise_df['time'], frac=0.2)
    #plt.plot(guess_exercise_df['time'], loess_smoothed[:, 1], 'r-')
    
    cols = list(exercise_labelled_df.columns.values)
    del cols[0]
    
    X = exercise_labelled_df[cols].values
    y = exercise_labelled_df["action"].tolist()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    model = make_pipeline(
         #GaussianNB()
         KNeighborsClassifier(n_neighbors=3)
    )
    
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))
    
    time_spent_action = 2 #This is the number of seconds we look at each increment of the workout
    workout_actions = get_workout_actions(time_spent_action,workout_df,model)
    elapsed_time = workout_df["time"].iloc[-1] 
    print_workout_stats(workout_actions,elapsed_time)
    print_timeline(workout_actions,time_spent_action,elapsed_time)
    

if __name__=='__main__':
    exercise_labelled = "exercise-model-data.csv"
    workout = sys.argv[1]
    main(exercise_labelled,workout)
