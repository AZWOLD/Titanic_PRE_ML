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

#Cleaning/Sorting Test DataFrame:
TS_Data_df = TS_Data_df.drop(columns=["Name","Ticket"])
TS_Data_df["Sex"] = TS_Data_df["Sex"].map({"male":0,"female":1})
TS_Data_df["Embarked"] = TS_Data_df["Embarked"].map({"S":0,"C":1,"Q":2})
TS_Data_df["Age"].fillna(int(TS_Data_df["Age"].mean()),inplace=True)
TS_Data_df["Fare"].fillna(int(TS_Data_df["Fare"].mean()),inplace=True)
TS_Data_df["Cabin"] = TS_Data_df["Cabin"].notna().astype(int)

#Reshaping data for Fitting and predicting:
X_Data = np.array(TR_Data_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin"]]).reshape(891,7)
Y_Data = np.array(TR_Data_df["Survived"])
X_TS_Data = np.array(TS_Data_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin"]]).reshape(418,7)

#Loading the model from Sklearn and fitting it with the data:
LR_model = linear_model.LogisticRegression()
LR_model.fit(X_Data,Y_Data)

#Predicting and Exporting Result File:
PRE_Y_Data = LR_model.predict(X_TS_Data)
TS_Data_df["Survived"] = PRE_Y_Data
TS_Data_df.to_csv("Prediction_Results.csv",index=False)
