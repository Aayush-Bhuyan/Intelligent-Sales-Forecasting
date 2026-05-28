import streamlit as st
st.title("Intelligent Sales Forecasting Dashboard")
st.write("Deployed analytics dashboard")


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import warnings
warnings.filterwarnings("ignore")

 
 


df = pd.read_csv("https://raw.githubusercontent.com/Aayush-Bhuyan/Intelligent-Sales-Forecasting/refs/heads/main/data/Adidas%20US%20Sales%20Datasets.csv")
df.head()


df.columns = df.columns.str.strip()
df.columns


df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], format="%d-%m-%Y")
df.dtypes

 
 


currency_cols = ["Price per Unit", "Total Sales", "Operating Profit"]
for col in currency_cols:
    df[col] = (df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False))
    df[col] = pd.to_numeric(df[col],errors='coerce')

df[currency_cols].head()

 



df["Year"] = df["Invoice Date"].dt.year
df["Month"] = df["Invoice Date"].dt.month
df["Quarter"] = df["Invoice Date"].dt.quarter
df["Month Name"] = df["Invoice Date"].dt.month_name()
df["Weekday"] = df["Invoice Date"].dt.day_name()

df.head()


print('Rows:', df.shape[0])
print('Columns:', df.shape[1])

df.info()

 



df.isnull().sum()

 



total_revenue = df['Total Sales'].sum()
total_profit = df['Operating Profit'].sum()
best_region = (df.groupby('Region')['Total Sales'].sum().idxmax())
best_product = (df.groupby('Product')['Total Sales'].sum().idxmax())

print('Total Revenue:', total_revenue)
print('Total Profit:', total_profit)
print('Best Region:', best_region)
print('Best Product:', best_product)

 



monthly_sales = (df.groupby('Invoice Date')['Total Sales'].sum().reset_index())
monthly_sales.head()

 
 


import streamlit as st

fig = px.line(monthly_sales, x='Invoice Date', y='Total Sales', title='Monthly Revenue Trend')
st.plotly_chart(fig, use_container_width=True)

 
 


region_sales = (df.groupby('Region')['Total Sales'].sum().reset_index())
fig = px.bar(region_sales, x='Region', y='Total Sales', color='Region', title='Revenue by Region')
st.plotly_chart(fig, use_container_width=True)

 



product_sales = (df.groupby('Product')['Total Sales'].sum().reset_index())
fig = px.bar(product_sales, x='Product', y='Total Sales', color='Product', title='Revenue by Product')
st.plotly_chart(fig, use_container_width=True)

 



channel_sales = (df.groupby('Sales Method')['Total Sales'].sum().reset_index())
fig = px.pie(channel_sales, names='Sales Method', values='Total Sales', title='Sales Channel Distribution')
st.plotly_chart(fig, use_container_width=True)

 
 


df['Profit Margin'] = (df['Operating Profit'] / df['Total Sales']) * 100
margin_analysis = (df.groupby('Product')['Profit Margin'].mean().reset_index())
fig = px.bar(margin_analysis, x='Product', y='Profit Margin', color='Profit Margin', title='Average Profit Margin by Product')
st.plotly_chart(fig, use_container_width=True)

 



heatmap_data = df.pivot_table(values='Total Sales', index='Region', columns='Month', aggfunc='sum')
fig = px.imshow(heatmap_data, text_auto=True, title='Regional Monthly Sales Heatmap')
st.plotly_chart(fig, use_container_width=True)

 


forecast_df = (monthly_sales.rename(columns={'Invoice Date': 'ds', 'Total Sales': 'y'}))
forecast_df.head()

 



model = Prophet()
model.fit(forecast_df)

 


future = model.make_future_dataframe(periods=6, freq='ME')
future.tail()

 
 


forecast = model.predict(future)
forecast.head()


fig1 = model.plot(forecast)
st.pyplot(fig1)

fig2 = model.plot_components(forecast)
st.pyplot(fig2)




actual = forecast_df['y']
predicted = forecast['yhat'][:len(actual)]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print('MAE:', mae)
print('RMSE:', rmse)




df.to_csv(
    "cleaned_adidas_sales.csv",
    index=False
)

print("Cleaned dataset saved.")
