import pandas as pd

from sklearn.linear_model import LogisticRegression

import joblib

train_data = pd.read_csv('data/train_data.csv')
x_vars = ["Age","Years","Num_Sites","Account_Manager"]
y_vars = ["Churn"]

X = train_data[x_vars]
y = train_data[y_vars]

model = LogisticRegression()
model.fit(X,y)

joblib.dump(model,'data/churn_model.pkl')






