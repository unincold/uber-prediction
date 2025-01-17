import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
pd.options.display.max_rows = None
pd.options.display.max_columns = None
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

data =pd.read_csv("rideshare_kaggle.csv")  #not prepared yet
df = data.copy()
df.dropna(subset=['price'],inplace = True)

#removing unneccessary columns
'''might be useful for the price prediction'''
df.drop(columns = ['id','latitude', 'longitude', 'product_id', 'timezone'], inplace =True)
df.drop(columns = ['short_summary', 'long_summary'], inplace =True)

#creating enviroment releated columns
enviroment_cols= ['temperature', 'apparentTemperature',
       'precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'windGustTime', 'visibility', 'temperatureHigh',
       'temperatureHighTime', 'temperatureLow', 'temperatureLowTime',
       'apparentTemperatureHigh', 'apparentTemperatureHighTime',
       'apparentTemperatureLow', 'apparentTemperatureLowTime',
       'dewPoint', 'pressure', 'windBearing', 'cloudCover', 'uvIndex',
       'visibility.1', 'ozone', 'sunriseTime', 'sunsetTime', 'moonPhase',
       'precipIntensityMax', 'uvIndexTime', 'temperatureMin',
       'temperatureMinTime', 'temperatureMax', 'temperatureMaxTime',
       'apparentTemperatureMin', 'apparentTemperatureMinTime',
       'apparentTemperatureMax', 'apparentTemperatureMaxTime', 'price']

df_enviroment_cols = df[enviroment_cols]
'''what we need to find is below'''
df.drop(columns= ['temperature', 'apparentTemperature',
       'precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'windGustTime', 'visibility', 'temperatureHigh',
       'temperatureHighTime', 'temperatureLow', 'temperatureLowTime',
       'apparentTemperatureHigh', 'apparentTemperatureHighTime',
       'apparentTemperatureLow', 'apparentTemperatureLowTime',
       'dewPoint', 'pressure', 'windBearing', 'cloudCover', 'uvIndex',
       'visibility.1', 'ozone', 'sunriseTime', 'sunsetTime', 'moonPhase',
       'precipIntensityMax', 'uvIndexTime', 'temperatureMin',
       'temperatureMinTime', 'temperatureMax', 'temperatureMaxTime',
       'apparentTemperatureMin', 'apparentTemperatureMinTime',
       'apparentTemperatureMax', 'apparentTemperatureMaxTime'], inplace =True)

df.drop(columns = ['timestamp','hour', 'day' ,'month','datetime'], inplace = True)
df_prep= pd.get_dummies(df, columns = ['icon','source' , 'destination','cab_type','name'])

#replacing categorical varaiable with numerical
df_prep = df_prep.replace({True:1,False:0})

#calculating iqr for price
q1 = df_prep['price'].quantile(.25)
q3 = df_prep['price'].quantile(.75)
iqr = q3-q1
#price outlier treatment
df_prep = df_prep[df_prep['price'] < (q3+1.5*iqr)]
#calculating iqr for distance
q1 = df_prep['distance'].quantile(.25)
q3 = df_prep['distance'].quantile(.75)
iqr = q3-q1
#distance outlier treatment
df_prep = df_prep[df_prep['distance'] < (q3+1.5*iqr)]
#defining independent and dependent feature
x = df_prep.drop(columns = ['price'])
y = df_prep['price']
ss = StandardScaler()
#standardizing the data
x = ss.fit_transform(x)
#splitting in train and test
x_train, x_test, y_train,y_test=train_test_split(x,y, train_size=0.2)
#model preperation
linear = LinearRegression()
#model_fitting
linear.fit(x_train, y_train)

y_pred = linear.predict(x_test)
# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

rf = RandomForestRegressor()
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")