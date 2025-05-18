#Importing needed libraries:
import pandas as pd

#Loading data from a csv file(using Pandas):
Data_df = pd.read_csv("train.csv")

#Basically sorting/cleaniing the DataFrame for usability:
Data_df = Data_df.set_index("PassengerId").drop(columns=["Name"])
Data_df["Sex"] = Data_df["Sex"].map({"male": 1,"female":0})
Data_df["Survived"] = Data_df.pop("Survived")
