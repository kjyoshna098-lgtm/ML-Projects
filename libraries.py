import numpy as np 
import pandas as pd 
import re
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/kjyos/OneDrive/Desktop/Amazon datasets.csv')
df = pd.DataFrame(data)
print (df.head(3))
print(df.info())
print(df.describe())
df.dropna(inplace = True)
duplicates = df[df.duplicated(keep=False)]
correlation = df.select_dtypes(include = ['number']).corr()
correlation
df.groupby("Category")["TotalAmount"].sum()
df.groupby("ProductName")["Discount"].mean()
df.groupby("Country")["TotalAmount"].sum()
df[df["Discount"] > 20]
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["DiscountRate"] = df["Discount"] / df["UnitPrice"]
df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True)
df["Month"] = df["OrderDate"].dt.month
df["Year"] = df["OrderDate"].dt.year
from sklearn.cluster import KMeans
X = df[["Quantity","UnitPrice","Discount"]]
kmeans = KMeans(n_clusters=4)
df["CustomerCluster"] = kmeans.fit_predict(X)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = df[["Quantity","UnitPrice","Tax","ShippingCost"]]
y = df["Discount"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train,y_train)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X = df[["Quantity","UnitPrice","Tax","ShippingCost"]]
y = df["Discount"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
pred = model.predict(X_test)
mean_squared_error(y_test,pred)