import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

# Predicting the price of house based on the area,age,bedrooms:
df = pd.read_csv("HousePrediction.csv")
print(df)
# data pre-processing:
# Filling the Null values using the median of the column
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

# linear regression:
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
# there is 3 coefficient because of 3 variables which is called (m1 = area, m2 = bedrooms, m3 = age):
print(reg.coef_)
# there is 1 coefficient because of 1 variable which is called (b = price):
print(reg.intercept_)
# it will give you the price of an house taking the 3000 area, 3 bedrooms, 40 age:
reg_predict = reg.predict([[3000, 3, 40]])
print(reg_predict)
reg_predict_1 = reg.predict([[2500, 4, 5]])
print(reg_predict_1)

'''
# Predicting the salary of employs using the test_score, experience, interview_score:
df = pd.read_csv('Hiring.csv')
print(df)
# data pre-processing:
# Filling the Null values using the median of the column
median_test_score = math.floor(df.test_score.median())
print(median_test_score)
df.test_score = df.test_score.fillna(median_test_score)
print(df)

# linear regression:
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']], df.salary)
# there is 3 coefficient because of 3 variables which is called (m1 = experience, m2 = test_score, m3 = interview_score):
print(reg.coef_)
# there is 1 coefficient because of 1 variable which is called (b = salary):
print(reg.intercept_)
# it will give you the price of an house taking the 3000 area, 3 bedrooms, 40 age:
reg_predict = reg.predict([[2, 9, 6]])
print(reg_predict)
reg_predict_1 = reg.predict([[12, 10, 10]])
print(reg_predict_1)
'''