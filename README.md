Background:
Sensors produce autonomous data which can be collected continuously or periodically. Sensors work like a minicomputer with its own set of operations. Working with sensor data will help to automate actions. 

Purpose:
The purpose of this assignment is to learn how to analyse and forecast sensor data. Sensors produce data which can be collected over a period. This historical data can be utilized to analyse and perform predictions. 

ARIMA:
Autoregressive (AR): This component uses lagged values of the time series to predict future values. The number of lagged values used is represented by the parameter 'p' in the ARIMA model.
Integrated (I): This part involves differencing the time series to achieve stationarity. The degree of differencing is denoted by 'd'. A stationary time series is one whose statistical properties like mean and variance do not change over time.
Moving Average (MA): The MA component models the error of the model as a combination of previous error terms. The parameter 'q' represents the size of the moving average window.


1.analyse multiple columns of the provided dataset. Show scatterplots and correlation matrices of chosen columns. Explain what you understood.
2.Use “adfuller” to understand if  data is stationary
3.Use “auto_arima” to find the best ARIMA model parameters
4.Forecasting: Train and test CO2 data with 70% training and 30% test. 
5.Plot forecast with actual values as shown in the below figure. Use stepwise approach for better forecast values.
6.Show MAE value of the test
