## 📚 Table of Contents

- [🛥️ Description](#-description)
- [🚀 Setup](#-setup)
- [🤖 Overview](#-Overveiw)
  - [🔧 Want to test your own data?](#-want-to-test-your-own-data)
- [🏃 Running the ML](#-running-the-ml)
- [🙏 Credit](#-credit)

# 🛥️ Description

This is a simple machine learning project using the famous Kaggle dataset: **Titanic - Machine Learning from Disaster**.

The model is trained to predict whether a passenger survived or not based on their details (such as `PassengerId`, `Ticket`, `Cabin`, `Embarked`, etc).  
The current model accuracy is approximately **77%**.

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/AZWOLD/Titanic_PRE_AI.git
```
### 2. Navigate to the folder
```bash
cd Titanic_PRE_AI.py
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🤖 Overview
The ML predicts the survival stat for all the passengers in the "**test.csv**" file.

>### 🔧 Want to test your own data?
>Edit the test.csv file and add your custom passengers. Then run the script as shown below.

## 🏃 Running the ML
Open your Terminal or command prompt inside the project folder and run
```bash
python Titanic_PRE_AI.py
```
The script will output the survival predictions based on the **test.csv** content.

## 🙏 Credit
Created with 💻 by [AZWOLD](https://github.com/AZWOLD)
