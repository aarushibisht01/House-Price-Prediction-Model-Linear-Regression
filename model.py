import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#loading the dataset
df=pd.read_csv("data.csv")

#Exploratory Data Analysis

print("First 5 rows of dataset:\n")
print(df.head())

print("Last 5 rows of dataset:\n")
print(df.tail())

print("Random 5 rows of dataset:\n")
print(df.sample(5))

print("Rows and columns of dataset:\n")
print(df.shape)

print("Column names of dataet:\n")
print(df.columns)

print("Datatypes of each column:\n")
print(df.info())

print("Statistical summary of dataset:\n")

print("For numerical columns:\n")
print(df.describe())

print("For categorical columns:\n")
print(df.describe(include="object"))

print("Number of missing values:\n")
print(df.isnull().sum().sum())