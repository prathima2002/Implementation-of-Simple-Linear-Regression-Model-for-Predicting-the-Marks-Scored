# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program
2.Import the numpy,pandas,matplotlib
3.Read the dataset of student scores
4.Assign the columns hours to x and columns scores to y
5.From sklearn library select the model to train and test the dataset
6.Plot the training set and testing set in the graph using matplotlib library
7.Stop the program
```   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: prathima
RegisterNumber: 212220040156
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values #assigning colum hours to X
X  
Y=dataset.iloc[:,1].values 
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
<img width="287" alt="image" src="https://user-images.githubusercontent.com/108709865/192093058-1454e00f-6411-4362-ad28-511e58167389.png">
<img width="328" alt="image" src="https://user-images.githubusercontent.com/108709865/192093076-eacd6b66-25d3-49a2-89eb-9c0134bba596.png">



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
