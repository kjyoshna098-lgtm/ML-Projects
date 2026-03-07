import pandas as pd
from sklearn.cluster import KMeans
df=pd.read_csv('C:/Users/kjyos/OneDrive/Desktop/Amazon datasets.csv')
X = df[["Quantity","TotalAmount","Discount"]]
kmeans = KMeans(n_clusters=4,random_state=0)
df["Cluster"] = kmeans.fit_predict(X)
print(df["Cluster"].value_counts())