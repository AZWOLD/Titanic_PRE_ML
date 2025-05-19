#Importing needed libraries:
import pandas as pd
import numpy as np
from sklearn import linear_model

#Loading data from a csv file(using Pandas):
Data_df = pd.read_csv("train.csv")

#Basically sorting/cleaniing the DataFrame for usability:
Data_df = Data_df.set_index("PassengerId").drop(columns=["Name"])
Data_df["Sex"] = Data_df["Sex"].map({"male": 1,"female":0})
Data_df["Embarked"] = Data_df["Embarked"].map({"S": 0,"C":1,"Q":2})
Data_df["Survived"] = Data_df.pop("Survived")
Data_df["Age"].fillna(int(Data_df["Age"].mean()),inplace=True)
Data_df["Cabin"] = Data_df["Cabin"].notna().astype(int)

#Reshaping data for Inputing:
X_Data = np.array(Data_df[["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin"]]).reshape(-1,1)
Y_Data = np.array(Data_df["Survived"])

#Loading the model from Sklearn and fitting it with the data:
LR_model = linear_model.LogisticRegression()
LR_model.fit(X_Data,Y_Data)