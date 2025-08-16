import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Dataset/test.csv")
labels = np.unique(dataset['label'])
dataset.drop(['Timestamp'], axis = 1,inplace=True)
le = LabelEncoder()
dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))#encode all str columns to numeric
dataset.fillna(dataset.mean(), inplace = True)
Y = dataset['label'].ravel()
dataset.drop(['label'], axis = 1,inplace=True)
print(np.unique(Y, return_counts=True))
X = dataset.values
print(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) #split dataset into train and test
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
predict = dt.predict(X_test)
acc = accuracy_score(y_test, predict)
mse = mean_squared_error(y_test, predict)
print(acc)
print(mse)
