#this code is to import libraries
import numpy as np

# this code is to import pandas
import pandas as pd

# this code is to import matplot lid library
import matplotlib.pyplot as plt

# this code is to create a variable to store the dataset
dataset= pd.read_csv("Salary_Data.csv")

A=dataset.iloc[:,0:2]
B=dataset.iloc[:,1:2]

#spliting the dataset into train data and test data
from sklearn.model_selection import train_test_split

#creating variable to store A_train,A_test and B_train,B_test
A_train,A_test,B_train,B_test=train_test_split(A,B,test_size=1/3,random_state=0)

#training the decision LinearRegression
from sklearn.linear_model import LinearRegression

#creating variable and assigning the LinearRegression algorithm
dataset_module=LinearRegression

#training the module dataset with A_train and B_train
dataset_module.fit(A_train.B_train)
#prediction making
dataset_prediction=dataset_module.predict(A_test)

dataset_prediction


