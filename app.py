import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# loading the dataset
cal=fetch_california_housing()
df=pd.DataFrame(cal.data, columns=cal.feature_names)
df['Price']=cal.target

# title of the app
st.title("California House Price Prediction App")

# Data Overview
st.write("### Data Overview")
st.write(df.head(10))

# split the dataset
X=df.drop('Price', axis=1)
y=df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sc= StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Model_selection
st.write("select the model")
model_name=st.selectbox("Choose the model", ["Lin_Reg", 'Rd', 'Ls'])

# Initalize the model
models={
    'Lin_Reg':LinearRegression(),
    'Rd':Ridge(),
    'Ls':Lasso(alpha=0.01)
    }
# train the model
mod=models[model_name]
mod.fit(X_train_sc, y_train)

y_pred=models[model_name].predict(X_test_sc)
# Display evaluation metrics
test_mse=mean_squared_error(y_test, y_pred)
test_r2=r2_score(y_test, y_pred)

st.write("### Evaluation Metrics")
st.write("Test MSE: ", test_mse)
st.write("Test R2: ", test_r2)

# features inputs by user
st.write("### Select the features")
user_input={}
for feature in X.columns:
    user_input[feature]=st.number_input(feature, value=float(X[feature].mean()))

user_input_df=pd.DataFrame([user_input])
user_input_sc=sc.transform(user_input_df)

# predict the house price
prediction=mod.predict(user_input_sc)
st.write("### Predicted House Price")
st.write(f"Predicted price for the inputs: {prediction[0]}")


