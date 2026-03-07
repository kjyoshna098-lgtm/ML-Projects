import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv('C:/Users/kjyos/OneDrive/Desktop/Amazon datasets.csv')
X = df[["Quantity","UnitPrice","ShippingCost","Tax"]]
y = df["Discount"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)
model.fit(X_train,y_train)
pred = model.predict(X_test)
rmse = mean_squared_error(y_test,pred)**0.5
print("RMSE:",rmse)