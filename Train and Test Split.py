#Imports
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd


#Functions
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder '{folder_name}' already exists.")
        
#Global Variables
PREPROCESSED_FOLDER = Path("Datasets/Preprocessed Data")
TRAINING_FOLDER = Path("Datasets/Training Data")
TESTING_FOLDER = Path("Datasets/Testing Data")

create_folder(TRAINING_FOLDER)
create_folder(TESTING_FOLDER)
df = pd.read_csv(PREPROCESSED_FOLDER / "fixed_data.csv")


#      _______        _   _                 _________        _       _                _____       _ _ _   
#     |__   __|      | | (_)               / /__   __|      (_)     (_)              / ____|     | (_) |  
#        | | ___  ___| |_ _ _ __   __ _   / /   | |_ __ __ _ _ _ __  _ _ __   __ _  | (___  _ __ | |_| |_ 
#        | |/ _ \/ __| __| | '_ \ / _` | / /    | | '__/ _` | | '_ \| | '_ \ / _` |  \___ \| '_ \| | | __|
#        | |  __/\__ \ |_| | | | | (_| |/ /     | | | | (_| | | | | | | | | | (_| |  ____) | |_) | | | |_ 
#        |_|\___||___/\__|_|_| |_|\__, /_/      |_|_|  \__,_|_|_| |_|_|_| |_|\__, | |_____/| .__/|_|_|\__|
#                                  __/ |                                      __/ |        | |            
#                                 |___/                                      |___/         |_|            
x = df.drop(columns=["Target"])
y = df["Target"]

#Splits up the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_data = x_train.copy()
train_data["Target"] = y_train

test_data = x_test.copy()
test_data["Target"] = y_test

train_data.to_csv(TRAINING_FOLDER / "training data.csv", index=False)
test_data.to_csv(TESTING_FOLDER / "testing data.csv", index=False)

print("Original dataset:", df.shape)
print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)


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
print("Baseline Accuracy:", accuracy, "%")