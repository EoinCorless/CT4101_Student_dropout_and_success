print("\n\n\n"
      "Starting Script!"
        "\n\n\n")

#Imports
import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Functions
def check_for_folder(folder_name):
    if not os.path.exists(folder_name):
        print("Missing ",folder_name,"folder, Please run Train and Test Split.py")
        sys.exit() #Exits if folder not found

        
#Global Variables
PREPROCESSED_FOLDER = Path("Datasets/Preprocessed Data")
TRAINING_FOLDER = Path("Datasets/Training Data")
TESTING_FOLDER = Path("Datasets/Testing Data")

check_for_folder(TRAINING_FOLDER)
check_for_folder(TESTING_FOLDER)
df = pd.read_csv(PREPROCESSED_FOLDER / "fixed_data.csv")


#      _______        _       _             
#     |__   __|      (_)     (_)            
#        | |_ __ __ _ _ _ __  _ _ __   __ _ 
#        | | '__/ _` | | '_ \| | '_ \ / _` |
#        | | | | (_| | | | | | | | | | (_| |
#        |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
#                                      __/ |
#                                     |___/ 

#Deliminator is now "," after preprocessing, used to be ";"
train_df = pd.read_csv(TRAINING_FOLDER / "training data.csv", sep=",")
test_df = pd.read_csv(TESTING_FOLDER / "testing data.csv", sep=",")

#Sanity Check
print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)
print("\n")

#Training data
x_train = train_df.drop("Target", axis=1) #Axis 1 to operate on columns, not rows
y_train = train_df["Target"]

#Testing data
x_test = test_df.drop("Target", axis=1) #Axis 1 to operate on columns, not rows
y_test = test_df["Target"]

from sklearn.ensemble import RandomForestClassifier

#PLACEHOLDER VALUES
RandomForest_model = RandomForestClassifier(
    n_estimators=1, #Number of Decision trees
    max_depth=1, #Depth of EACH decision trees
    random_state=1, #Essentially a seed, doesn't matter what this value is as long as its consistent between models/runs
    class_weight="balanced" #Balances the importance of each target category
)

RandomForest_model.fit(x_train, y_train)#Train baseline model
y_pred = RandomForest_model.predict(x_test)#Generate predictions made


accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
print("Accuracy:", accuracy, "%")
print(classification_report(y_test, y_pred))