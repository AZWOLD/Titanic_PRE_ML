#Importing needed libraries:
import pandas as pd
import numpy as np
from sklearn import linear_model

#Loading data from a csv file(using Pandas):
TR_Data_df = pd.read_csv("train.csv")
TS_Data_df = pd.read_csv("test.csv")

#Basically sorting/cleaniing the DataFrame for usability:
TR_Data_df = TR_Data_df.set_index("PassengerId").drop(columns=["Name","Ticket"])
TR_Data_df["Sex"] = TR_Data_df["Sex"].map({"male": 1,"female":0})
TR_Data_df["Embarked"] = TR_Data_df["Embarked"].map({"S": 0,"C":1,"Q":2})
TR_Data_df["Survived"] = TR_Data_df.pop("Survived")
TR_Data_df["Age"].fillna(int(TR_Data_df["Age"].mean()),inplace=True)
TR_Data_df["Cabin"] = TR_Data_df["Cabin"].notna().astype(int)

#Reshaping data for Inputing:
X_Data = np.array(TR_Data_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin"]]).reshape(891,7)
Y_Data = np.array(TR_Data_df["Survived"])

#Loading the model from Sklearn and fitting it with the data:
LR_model = linear_model.LogisticRegression()
LR_model.fit(X_Data,Y_Data)

