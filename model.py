import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

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

print("Value counts of countries:\n")
print(df["country"].value_counts())

#Data Cleaning

#removing duplicate rows
df=df.drop_duplicates()

#removing missing values
df=df.dropna()
df = df[df["price"] > 0]

#Data Preprocessing

#removing date column
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df.drop("date", axis=1, inplace=True)
df.drop("street", axis=1, inplace=True)
df.drop("country", axis=1, inplace=True)

#seperating features and target
X=df.drop(columns="price",axis=1)
y=df["price"]

#encoding categorical features for data preprocessing
column_transformer=ColumnTransformer(
    transformers=[("encoding",OneHotEncoder(sparse_output=False),["city","statezip"])],
    remainder="passthrough"
)

X=column_transformer.fit_transform(X)

#training the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#feature scaling to normalize the range of independent variables
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#applying linear regression
model=LinearRegression()
model.fit(X_train,y_train)

#prediction output
y_predicted=model.predict(X_test)

#comparing actual vs predicted output
comparison_df=pd.DataFrame(
    {
        "Actual price":y_test,
        "Predicted price":y_predicted
    }
)
print(comparison_df.head())

#evaluating the model
mae=mean_absolute_error(y_test,y_predicted)
mse=mean_squared_error(y_test,y_predicted)
rmse=mse**0.5
r2=r2_score(y_test,y_predicted)

print(f"Mean aboslute error is {mae}")
print(f"Mean squeared error is {mse}")
print(f"Root mean squared error is {rmse}")
print(f"R2 score is {r2}")

#graphs

#scatterplot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predicted, alpha=0.6,marker='o')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_pridicted_prices.png", dpi=300, bbox_inches="tight")
plt.show()

#correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

#residual graph
residuals = y_test - y_predicted
plt.figure(figsize=(8,6))
sns.scatterplot( x=y_predicted,y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted House Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()