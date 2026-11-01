#Imports
from pathlib import Path
import pandas as pd
import os

#Global Variables
AUDIT_FOLDER = Path("Datasets/Preprocessed Data")
RAW_DATA_PATH = Path("Datasets/Raw Data/data.csv")

#Functions
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder '{folder_name}' already exists.")
        
#Creates output folder
create_folder(AUDIT_FOLDER)

#      _____                                            _               _____ _            _ _            
#     |  __ \                                          (_)             |  __ (_)          | (_)           
#     | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _  | |__) | _ __   ___| |_ _ __   ___ 
#     |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` | |  ___/ | '_ \ / _ \ | | '_ \ / _ \
#     | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| | | |   | | |_) |  __/ | | | | |  __/
#     |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, | |_|   |_| .__/ \___|_|_|_| |_|\___|
#                    | |                                         __/ |         | |                        
#                    |_|                                        |___/          |_|

#Reads and analyzes the raw data
df = pd.read_csv(RAW_DATA_PATH)
print("Shape before fixing delimiter:", df.shape)

#Fix delimiter being ; and not , for consistency (The pandas default delimiter is ,)   
df = pd.read_csv(RAW_DATA_PATH, sep=";")

#Tidies up column name (e.g. Portuguese typo)
df = df.rename(columns={"Nacionality": "Nationality"})
df = df.rename(columns={"Daytime/evening attendance	": "Daytime/evening attendance"})
df = df.rename(columns={"Nacionality": "Nationality"})
df = df.rename(columns={"Nacionality": "Nationality"})
df = df.rename(columns={"Nacionality": "Nationality"})
df.to_csv(AUDIT_FOLDER / "fixed_data.csv", index=False)
shape = df.shape #Expected output: (4424, 37)
print("Shape after fixing delimiter:", shape)

#Check for missing values
missing_values_num = df.isnull().sum()
missing_values_percentage = (df.isnull().mean() * 100).round(3)

missing_value_summary = pd.DataFrame({
    'Missing Values': missing_values_num,
    'Percentage': missing_values_percentage
})
if (missing_values_num.sum() <= 0):
    print("\nNo missing values found.")
else:
    print("\nMissing Values\n")
    print(missing_value_summary[missing_value_summary['Missing Values'] > 0])

#Check for duplicate rows
duplicates = df.duplicated().sum()
print("Number of Duplicate Rows:", duplicates)

#Check column data types
print("\nColumn Data Types:\n")
print(df.dtypes)

### Column value quality checks ###
print("\n")

#Gender
invalid_rows = df[~df["Gender"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Gender values:")
    print(invalid_rows)
else:
    print("No invalid Gender values found.")
    

#Age at enrollment
age_minimum = df["Age at enrollment"].min()
age_maximum = df["Age at enrollment"].max()
print("Minimum/Maximum Age at enrollment:", age_minimum,"/",age_maximum)

    
#Marital status
invalid_rows = df[~df["Marital status"].isin([1, 2, 3, 4, 5, 6])]
if(not invalid_rows.empty):
    print("Rows with invalid Marital status values:\n")
    print(invalid_rows)
else:
    print("No invalid Marital status values found.\n")


#Nationality
invalid_rows = df[~df["Nationality"].isin([1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109])]
if(not invalid_rows.empty):
    print("Rows with invalid Nationality values:")
    print(invalid_rows)
else:
    print("No invalid Nationality values found.")
    
    
#Displaced
invalid_rows = df[~df["Displaced"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Displaced values:")
    print(invalid_rows)
else:
    print("No invalid Displaced values found.")
    
    
#Educational special needs
invalid_rows = df[~df["Educational special needs"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Educational special needs values:\n")
    print(invalid_rows)
else:
    print("No invalid Educational special needs values found.\n")
    
    
#International
invalid_rows = df[~df["International"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid International values:")
    print(invalid_rows)
else:
    print("No invalid International values found.")
    
    
#Father's occupation
invalid_rows = df[~df["Father's occupation"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195])]
if(not invalid_rows.empty):
    print("Rows with invalid Father's occupation values:")
    print(invalid_rows)
else:
    print("No invalid Father's occupation values found.")
    
    
#Mother's occupation
invalid_rows = df[~df["Mother's occupation"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194])]
if(not invalid_rows.empty):
    print("Rows with invalid Mother's occupation values:\n")
    print(invalid_rows)
else:
    print("No invalid Mother's occupation values found.\n")
    
    
#Father's qualification
invalid_rows = df[~df["Father's qualification"].isin([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])]
if(not invalid_rows.empty):
    print("Rows with invalid Father's qualification values:")
    print(invalid_rows)
else:
    print("No invalid Father's qualification values found.")
    
    
#Mother's qualification
invalid_rows = df[~df["Mother's qualification"].isin([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])]
if(not invalid_rows.empty):
    print("Rows with invalid Mother's qualification values:")
    print(invalid_rows)
else:
    print("No invalid Mother's qualification values found.")
    
    
#Scholarship holder
invalid_rows = df[~df["Scholarship holder"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Scholarship holder values:\n")
    print(invalid_rows)
else:
    print("No invalid Scholarship holder values found.\n")
    
    
#Tuition fees up to date
invalid_rows = df[~df["Tuition fees up to date"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Tuition fees up to date values:")
    print(invalid_rows)
else:
    print("No invalid Tuition fees up to date values found.")
    
    
#Debtor
invalid_rows = df[~df["Debtor"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Debtor values:")
    print(invalid_rows)
else:
    print("No invalid Debtor values found.")
    
    
#Course
invalid_rows = df[~df["Course"].isin([33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991])]
if(not invalid_rows.empty):
    print("Rows with invalid Course values:\n")
    print(invalid_rows)
else:
    print("No invalid Course values found.\n")
    
    
#Admission Grade
invalid_rows = df[~df["Admission grade"].between(0, 200)]
if(not invalid_rows.empty):
    print("Rows with invalid Admission Grade values:")
    print(invalid_rows)
else:
    print("No invalid Admission Grade values found.")
    
    
#Application mode
invalid_rows = df[~df["Application mode"].isin([1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57])]
if(not invalid_rows.empty):
    print("Rows with invalid Application mode values:")
    print(invalid_rows)
else:
    print("No invalid Application mode values found.")
    
    
#Application Order
invalid_rows = df[~df["Application order"].between(0, 9)]
if(not invalid_rows.empty):
    print("Rows with invalid Application Order values:\n")
    print(invalid_rows)
else:
    print("No invalid Application Order values found.\n")
    
    
#Daytime/evening attendance
invalid_rows = df[~df["Daytime/evening attendance"].isin([0, 1])]
if(not invalid_rows.empty):
    print("Rows with invalid Daytime/evening attendance values:")
    print(invalid_rows)
else:
    print("No invalid Daytime/evening attendance values found.")
    
    
#Previous qualification
invalid_rows = df[~df["Previous qualification"].isin([1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43])]
if(not invalid_rows.empty):
    print("Rows with invalid Previous qualification values:")
    print(invalid_rows)
else:
    print("No invalid Previous qualification values found.")
    
    
#Previous qualification (grade)
invalid_rows = df[~df["Previous qualification (grade)"].between(0, 200)]
if(not invalid_rows.empty):
    print("Rows with invalid Previous qualification (grade) values:")
    print(invalid_rows,"\n")
else:
    print("No invalid Previous qualification (grade) values found.\n")
    

# - #

#Curricular units 1st sem (credited)
CU1C_minimum = df["Curricular units 1st sem (credited)"].min()
CU1C_maximum = df["Curricular units 1st sem (credited)"].max()
print("Minimum/Maximum Curricular units 1st sem (credited):", CU1C_minimum,"/", CU1C_maximum)

#Curricular units 1st sem (enrolled)
CU1EN_minimum = df["Curricular units 1st sem (enrolled)"].min()
CU1EN_maximum = df["Curricular units 1st sem (enrolled)"].max()
print("Minimum/Maximum Curricular units 1st sem (enrolled):", CU1EN_minimum,"/", CU1EN_maximum)

#Curricular units 1st sem (evaluations)
CU1EV_minimum = df["Curricular units 1st sem (evaluations)"].min()
CU1EV_maximum = df["Curricular units 1st sem (evaluations)"].max()
print("Minimum/Maximum Curricular units 1st sem (evaluations):", CU1EV_minimum,"/", CU1EV_maximum)

#Curricular units 1st sem (approved)
CU1A_minimum = df["Curricular units 1st sem (approved)"].min()
CU1A_maximum = df["Curricular units 1st sem (approved)"].max()
print("Minimum/Maximum Curricular units 1st sem (approved):", CU1A_minimum,"/", CU1A_maximum)

#Curricular units 1st sem (grade)
invalid_rows = df[~df["Curricular units 1st sem (grade)"].between(0, 200)]
if(not invalid_rows.empty):
    print("Rows with invalid Curricular units 1st sem (grade) values:")
    print(invalid_rows)
else:
    print("No invalid Curricular units 1st sem (grade) values found.")
    

#Curricular units 1st sem (without evaluations)
CU1WEV_minimum = df["Curricular units 1st sem (without evaluations)"].min()
CU1WEV_maximum = df["Curricular units 1st sem (without evaluations)"].max()
print("Minimum/Maximum Curricular units 1st sem (without evaluations):", CU1WEV_minimum,"/", CU1WEV_maximum)

# - #
print("\n")

#Curricular units 2nd sem (credited)
CU2C_minimum = df["Curricular units 2nd sem (credited)"].min()
CU2C_maximum = df["Curricular units 2nd sem (credited)"].max()
print("Minimum/Maximum Curricular units 2nd sem (credited):", CU2C_minimum,"/", CU2C_maximum)

#Curricular units 2nd sem (enrolled)
CU2EN_minimum = df["Curricular units 2nd sem (enrolled)"].min()
CU2EN_maximum = df["Curricular units 2nd sem (enrolled)"].max()
print("Minimum/Maximum Curricular units 2nd sem (enrolled):", CU2EN_minimum,"/", CU2EN_maximum)

#Curricular units 2nd sem (evaluations)
CU2EV_minimum = df["Curricular units 2nd sem (evaluations)"].min()
CU2EV_maximum = df["Curricular units 2nd sem (evaluations)"].max()
print("Minimum/Maximum Curricular units 2nd sem (evaluations):", CU2EV_minimum,"/", CU2EV_maximum)

#Curricular units 2nd sem (approved)
CU2A_minimum = df["Curricular units 2nd sem (approved)"].min()
CU2A_maximum = df["Curricular units 2nd sem (approved)"].max()
print("Minimum/Maximum Curricular units 2nd sem (approved):", CU2A_minimum,"/", CU2A_maximum)

#Curricular units 2nd sem (grade)
invalid_rows = df[~df["Curricular units 2nd sem (grade)"].between(0, 200)]
if(not invalid_rows.empty):
    print("Rows with invalid Curricular units 2nd sem (grade) values:")
    print(invalid_rows)
else:
    print("No invalid Curricular units 2nd sem (grade) values found.")
    

#Curricular units 2nd sem (without evaluations)
CU2WEV_minimum = df["Curricular units 2nd sem (without evaluations)"].min()
CU2WEV_maximum = df["Curricular units 2nd sem (without evaluations)"].max()
print("Minimum/Maximum Curricular units 2nd sem (without evaluations):", CU2WEV_minimum,"/", CU2WEV_maximum)

#Target
invalid_rows = df[~df["Target"].isin(["Graduate", "Dropout", "Enrolled"])]
if(not invalid_rows.empty):
    print("Rows with invalid Target values:")
    print(invalid_rows)
else:
    print("No invalid Target values found.")

graduate_count = df[df["Target"] == "Graduate"].shape[0]
dropout_count = df[df["Target"] == "Dropout"].shape[0]  
enrolled_count = df[df["Target"] == "Enrolled"].shape[0]
target_percentage = df["Target"].value_counts(normalize=True)*100

print("\nTarget Distribution\n")
print("Absolute Value\n",
      "Graduate:", graduate_count,"\n",
      "Dropout:", dropout_count,"\n",
      "Enrolled:", enrolled_count,"\n")
print("Percentages")
print(target_percentage)
print("\n")