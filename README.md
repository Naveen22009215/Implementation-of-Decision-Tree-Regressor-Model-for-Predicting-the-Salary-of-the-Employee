# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload the dataset in the compiler and read the dataset.

3. Find head,info and null elements in the dataset.

4. Using LabelEncoder and DecisionTreeRegressor , find MSE and R2 of the dataset.

5. Predict the values and end the program.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: P NAVEEN KUMAR
RegisterNumber: 212222230092 
*/
```
```

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```


## Output:
## 1. data.head()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/a961db6b-37dc-4b17-87ca-2b54ee74a0dd)

## 2. data.info()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/3d2f0f88-07e0-472d-bd9c-930a2b8bb464)

## 3. isnull() and sum()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/084c16e3-79b6-420d-b1ac-e5e48c92b140)

## 4. data.head() for salary
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/192f8f9f-bc86-47b2-9384-70bb1d558209)

## 5. MSE value
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/22279b98-6719-4168-8094-92de01c57abb)

## 6. r2 value
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/b8cfc6fc-d9a9-4bfe-9e61-7ba3ee4bf41f)

## 7. data prediction
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119401470/9f11d8db-ba75-41d3-a77c-4f0645220cb7)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
