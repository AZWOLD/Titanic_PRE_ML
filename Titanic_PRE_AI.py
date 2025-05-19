#Importing needed libraries:
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading data from a csv file(using Pandas):
TR_Data_df = pd.read_csv("train.csv")
TS_Data_df = pd.read_csv("test.csv")
#repeated variables:
Sex_Map = {"male": 1,"female":0}
Embarked_Map = {"S": 0,"C":1,"Q":2}
#Basically sorting/cleaniing the DataFrame for usability:
TR_Data_df = TR_Data_df.set_index("PassengerId").drop(columns=["Name","Ticket"])
TR_Data_df["Sex"] = TR_Data_df["Sex"].map(Sex_Map)
TR_Data_df["Embarked"] = TR_Data_df["Embarked"].map(Embarked_Map)
TR_Data_df["Survived"] = TR_Data_df.pop("Survived")
TR_Data_df["Age"].fillna(int(TR_Data_df["Age"].mean()),inplace=True)
TR_Data_df["Cabin"] = TR_Data_df["Cabin"].notna().astype(int)

#Cleaning/Sorting Test DataFrame:
TS_Data_df = TS_Data_df.drop(columns=["Name","Ticket"])
TS_Data_df["Sex"] = TS_Data_df["Sex"].map(Sex_Map)
TS_Data_df["Embarked"] = TS_Data_df["Embarked"].map(Embarked_Map)
TS_Data_df["Age"].fillna(int(TS_Data_df["Age"].mean()),inplace=True)
TS_Data_df["Fare"].fillna(int(TS_Data_df["Fare"].mean()),inplace=True)
TS_Data_df["Cabin"] = TS_Data_df["Cabin"].notna().astype(int)

#Reshaping data for Fitting and predicting:
X_Data = TR_Data_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin"]].values
Y_Data = TR_Data_df["Survived"].values
X_TS_Data = TS_Data_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin"]].values

#Loading the model from Sklearn and fitting it with the data:
LR_model = linear_model.LogisticRegression()
# LR_model.fit(X_Data,Y_Data)

#making a custom test data to test model accuracy:
X_train,X_val,Y_train,Y_val = train_test_split(X_Data,Y_Data,test_size=0.2,random_state=42)
#Testing the model:
LR_model.fit(X_train,Y_train)
preds = LR_model.predict(X_val)
print("Accuracy score: ",accuracy_score(Y_val,preds))
'''
#Predicting and Exporting Result File:
PRE_Y_Data = LR_model.predict(X_TS_Data)
TS_Data_df["Survived"] = PRE_Y_Data
TS_Data_df.to_csv("Prediction_Results.csv",index=False)
'''