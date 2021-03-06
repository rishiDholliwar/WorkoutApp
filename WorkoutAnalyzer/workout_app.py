import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
import time

"""
This program is used to analyze a workout.
It does this by taking in 2 arguments.
The first is a csv file containing the accelerometer data 
generated by Physics Toolbox Sensor Suite during a workout. 
Secondly, it takes in the user’s weight to get the calories burned.

The user will be presented information about their workout
First, the User is shown a Workout overview displaying the 
duration and the calories burned for the workout.
Second is the Workout Breakdown which prints the percentage and 
calories burned for each exercise.
Third is the Workout Timeline which shows 
a timeline of the exercise done during the workout.

Finally, the user is printed a graph of the acceleration data 
and a pie chart of the workout breakdown.

"""

"""
This function finds the actions the user took during their workout 
and returns them in order in a list. 
It works by looking at the workout data in small chunks (2-3 seconds) 
and then passing this small chunk of data to the model which predicts 
the action that the user took during this time.
"""
def get_workout_actions(seconds, df, model):
    workout_actions = []
    elapsed_time = df["time"].iloc[-1] 
    low = 1
    high = seconds
    
    while high < elapsed_time:
        subset_df =  df[((df.time >=low) & (df.time < high))]
        
        t_max_avg = subset_df.nlargest(5, 'aT (m/s^2)') 
        t_max_avg = t_max_avg['aT (m/s^2)'].mean()
        
        prediction = get_action(subset_df,model)
        
        low = high
        high += seconds
        
        workout_actions.append(prediction)
        
    return  workout_actions    
    
"""
Function is given a workout dataframe and a model
and returns the action predicted by the model
"""
def get_action(df, model):
    t_max_avg = df.nlargest(5, 'aT (m/s^2)') # may want to take more values
    t_max_avg = t_max_avg['aT (m/s^2)'].mean()
    
    predict_df = pd.DataFrame(columns=['t_max_avg'])
    predict_df = predict_df.append({'t_max_avg': t_max_avg }, ignore_index=True)

    return model.predict(predict_df)[0]

"""
Prints the Workout Overview and Breakdown
"""
def print_workout_stats(workout_actions, elapsed_time, weightKG):
    workout = Counter(workout_actions)
    numActions = len(workout_actions)
    total_cal_burned = 0
    workout_breakdown = ""
   
    for k,v in workout.items():
        percentage = str(round((v/numActions)*100,2))
        duration_action = (v/numActions)*elapsed_time
        
        calories_burned = get_calories_burned(duration_action, k, weightKG)
        total_cal_burned += calories_burned
        
        workout_breakdown += (" - " + k + ": " +  percentage + "%" + ", " 
                              + str(calories_burned) + " calories" + "\n")
        
    print("Workout Overview:")
    workout_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    print(" - Duration: " , workout_time + "\n - Calories Burned: ", round(total_cal_burned,2),"\n")
    print("Workout Breakdown:")
    print(workout_breakdown)
    return workout
    
   

"""
Function returns the calories burned for an action based the duration of the action 
and the users weight in kg

calories burned formula from:
https://www.hss.edu/conditions_burning-calories-with-exercise-calculating-estimated-energy-expenditure.asp
"""
def get_calories_burned(duration_action, action, weightKG):
    if action == 'walk':
        mets = 3.5
    elif action == 'jog':
        mets = 8
    elif action == 'run':
        mets = 12
    else:
        mets = 1.2 #Standing
    
    minutes = duration_action/60

    calories_burned = (.0175 * mets * float(weightKG)) * minutes
    return round(calories_burned, 2)
    
"""
Prints the time line of the workout
"""
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
            print("- Time ["+ str(prev_action_time) + "s-"+ str(current_action_time) + "s]", " = " + prev_action)
            prev_action_time = current_action_time
            prev_action = action
            
    print("- Time [" + str(prev_action_time) + "s-"+ str(elapsed_time) +"s]", " = " + action)
    
    
def main(exercise_labelled, workout, weightKG):
    exercise_labelled_df = pd.read_csv(exercise_labelled)
    workout_df = pd.read_csv(workout)
    
    plt.title("Workout")
    plt.ylabel("aT (m/s^2)")
    plt.xlabel("time (s)")
    plt.plot(workout_df['time'], workout_df['aT (m/s^2)'], 'b.', alpha=0.5)
    #loess_smoothed = lowess(workout_df['aT (m/s^2)'], workout_df['time'], frac=0.2)
    #plt.plot(workout_df['time'], loess_smoothed[:, 1], 'r-')
    
    cols = list(exercise_labelled_df.columns.values)
    del cols[0]
    
    X = exercise_labelled_df[cols].values
    y = exercise_labelled_df["action"].tolist()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    model = make_pipeline(
         KNeighborsClassifier(n_neighbors=3)
    )
    
    model.fit(X_train, y_train)
    #print(model.score(X_valid, y_valid))
    
    duration_action = 2 #This is the number of seconds we look at each increment of the workout
    workout_actions = get_workout_actions(duration_action, workout_df, model)
    elapsed_time = workout_df["time"].iloc[-1] 
    workout = print_workout_stats(workout_actions, elapsed_time, weightKG)
    print_timeline(workout_actions, duration_action, elapsed_time)
    
    plt.figure()
    plt.title("Workout Breakdown")
    plt.pie(workout.values(),  labels=workout.keys(), autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    

if __name__=='__main__':
    exercise_labelled = "exercise-model-data.csv"
    #workout = "test-data/workout7.csv" 
    workout = sys.argv[1]
    weightKG = sys.argv[2]
    main(exercise_labelled, workout, weightKG)
