# BU CS542 Machine Learning final project 
## Introduction
This is the final project of 2019 Spring Boston University CS542 Machine Learning.

### Team
Qingxing Li, Sichun Hao, Yuying Wang

### Project
Cryptocurrency Analysis and Prediction

### Data Source
https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory

### GOAL
Dig the in-depth information of the dataset, figure out the relationship within and between cryptocurrency. After that, Use LSTM to predict the future price of BitCoin and Etherum

### Repository Introduction
    -cryptocurrencypricehistory: all datasets
    -LSTMPrediction.py: python file for the LSTM prediction
    -ethereum_price.csv / bitcoin_price.csv: dataset used for prediction


### Prediction
Programming language: Python
TOOL: Tensorflow & Keras

In this part, we have used the Long Short Term Memory(LSTM) to do the prediction.
LSTM is a special RNN model which is very suitable for time series analysis and prediction.
By using the LSTM model given by Keras, a tool or API or tensorflow in python, achieved the 
prediction of the price of both Bitcoin and Ethereum from 2017-6-1 to 2018-2-20.



