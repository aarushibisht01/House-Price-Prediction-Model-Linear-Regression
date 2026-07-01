import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
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
print(df.isnull().sum())

#Data Cleaning

#removing duplicate rows
df=df.drop_duplicates()

#removing missing values
df=df.dropna()

#Data Preprocessing
#seperating features and target
X=df.drop(collumn="price",axis=1)
y=df["price"]

#encoding categorical features for data preprocessing
column_transformer=ColumnTransformer(
    transformers=[("encoding",OneHotEncoder(),["street","city","statezip","country"])],
    remaindr="passthrough"
)

X=column_transformer.fit_transform(X)

#training the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#feature scaling to normalize the range of independent variables
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
y_train=scaler.fit_transform(y_train)