print("\n\n\n"
      "Starting Script!"
        "\n\n\n")

#Imports
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

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
x_train = train_df.drop("Target", axis=1)
y_train = train_df["Target"]

#Testing data
x_test = test_df.drop("Target", axis=1)
y_test = test_df["Target"]

#      ____                 _ _              __  __           _      _ 
#     |  _ \               | (_)            |  \/  |         | |    | |
#     | |_) | __ _ ___  ___| |_ _ __   ___  | \  / | ___   __| | ___| |
#     |  _ < / _` / __|/ _ \ | | '_ \ / _ \ | |\/| |/ _ \ / _` |/ _ \ |
#     | |_) | (_| \__ \  __/ | | | | |  __/ | |  | | (_) | (_| |  __/ |
#     |____/ \__,_|___/\___|_|_|_| |_|\___| |_|  |_|\___/ \__,_|\___|_|
#                                                                                                                                          
baseline_model = DummyClassifier(strategy="most_frequent")#Create baseline model (guess most common reslt)
baseline_model.fit(x_train, y_train)#Train baseline model
y_pred = baseline_model.predict(x_test)#Generate predictions

accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
print("Accuracy:", accuracy, "%")
print(classification_report(y_test, y_pred, zero_division=0)) #Generates table of results (precision, recall, f1-score)
#classification_report causes "UndefinedMetricWarning" warning without zero_division=0 as a parameter