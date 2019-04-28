import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
#  training with LSTM
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.dates as mdates


def buildingset(cash_price,cin):

    # some parameters
    cash_price['Date_mpl'] = cash_price['Date'].apply(lambda x: mdates.date2num(x))
    split_date = '2017-06-01'
    window_len = 10
    epoch = 50
    batch_size = 2
    neurons = 50
    norm_cols = [metric for metric in ['Close','Volume']]
    kwargs = { 'close_off_high': lambda x: 2*(x['High']- x['Close'])/(x['High']-x['Low'])-1,
                'volatility': lambda x: (x['High']- x['Low'])/(x['Open'])}
    cash_price = cash_price.assign(**kwargs)


    model_data = cash_price[cash_price["Date"]>="2013-12-27"][['Date']+[metric for metric in ['Close','Volume','close_off_high','volatility']]]
    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    model_data.head()

    training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)


    # to build the training and testing input data set, using the window
    training_inputs = []
    for i in range(len(training_set)-window_len):
        temp_set = training_set[i:(i+window_len)].copy().astype("float")
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        training_inputs.append(temp_set)
    # training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1

    test_inputs = []
    for i in range(len(test_set)-window_len):
        temp_set = test_set[i:(i+window_len)].copy().astype("float")
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        test_inputs.append(temp_set)
    # test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1


    training_inputs = [np.array(training_input) for training_input in training_inputs]
    training_inputs = np.array(training_inputs)

    test_inputs = np.array([np.array(test_inputs) for test_inputs in test_inputs])

    # set a random seed to make sure generate same result
    np.random.seed(2)
    # initialise model architecture
    bitcoin_model = build_model(training_inputs, output_size=1, neurons=neurons)
    # model output is next price normalised to 10th previous closing price
    training_outputs = (training_set['Close'][window_len:].values / training_set['Close'][:-window_len].values) - 1
    # train model on data
    fit_result = bitcoin_model.fit(training_inputs, training_outputs,
                                        epochs=epoch, batch_size=batch_size, verbose=2, shuffle=True)
    # result
    visualization_result(fit_result,split_date,window_len,bitcoin_model,
                         training_set,model_data,training_inputs,test_set,test_inputs,coin)


def build_model(inputs, output_size, neurons, activ_func="selu",
                dropout=0.25, loss="mse", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


def visualization_result(bitcoin_history,split_date,window_len,bitcoin_model,
                         training_set,model_data,LSTM_training_inputs,test_set,LSTM_test_inputs,coin):

    # error
    plt.figure(1)
    plt.plot(bitcoin_history.epoch, bitcoin_history.history['loss'])
    title = "Training Error of "+coin
    plt.title(title)
    plt.ylabel('mse',fontsize=12)
    plt.xlabel('# Epochs',fontsize=12)
    plt.show(1)

    #training
    plt.figure(2)
    plt.xticks([datetime.date(i,j,1) for i in range(2013,2018) for j in [1,5,9]],
               [datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2018) for j in [1,5,9]])
    plt.plot(model_data[model_data['Date']< split_date]['Date'][window_len:],
             training_set['Close'][window_len:], label='Actual')
    plt.plot(model_data[model_data['Date']< split_date]['Date'][window_len:],
             ((np.transpose(bitcoin_model.predict(LSTM_training_inputs))+1) * training_set['Close'].values[:-window_len])[0],
             label='Predicted')
    title = "Training graph of "+coin
    plt.title(title)
    ylabel = "Price of "+coin
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.annotate('mse: %.4f'%np.mean(np.abs((np.transpose(bitcoin_model.predict(LSTM_training_inputs))+1)-\
                (training_set['Close'].values[window_len:])/(training_set['Close'].values[:-window_len]))),
                 xy=(0.75, 0.9),  xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')

    # figure inset code taken from http://akuederle.com/matplotlib-zoomed-up-inset
    # axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
    # axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2018) for j in [1,5,9]])
    # axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:],
    #          training_set['Close'][window_len:], label='Actual')
    # axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:],
    #          ((np.transpose(bitcoin_model.predict(LSTM_training_inputs))+1) * training_set['Close'].values[:-window_len])[0],
    #          label='Predicted')
    # axins.set_xlim([datetime.date(2016, 7, 1), datetime.date(2016, 9, 1)])
    # axins.set_ylim([500,750])
    # axins.set_xticklabels('')
    # mark_inset(ax1, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.show(2)

    # testing
    plt.figure(3)
    plt.xticks([datetime.date(j,i+1,1) for i in range(12) for j in range(2017,2019)],
               [datetime.date(j,i+1,1).strftime('%b %d %Y')  for i in range(12) for j in range(2017,2019)])
    plt.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:],
             test_set['Close'][window_len:], label='Actual')
    plt.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:],
             ((np.transpose(bitcoin_model.predict(LSTM_test_inputs))+1) * test_set['Close'].values[:-window_len])[0],
             label='Predicted')
    plt.annotate('mse: %.4f'%np.mean(np.abs((np.transpose(bitcoin_model.predict(LSTM_test_inputs))+1)-\
                (test_set['Close'].values[window_len:])/(test_set['Close'].values[:-window_len]))),
                 xy=(0.75, 0.9),  xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')
    title = "Testing graph of "+coin
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show(3)


def main():
    # import data
    color = sns.color_palette()
    cash_price = pd.read_csv("~/PycharmProjects/542project/bitcoin_price.csv", parse_dates=['Date'])
    cash_price.head()
    cash_price[cash_price["Date"]>='2013-12-27']['Volume'] = cash_price[cash_price["Date"]>='2013-12-27']["Volume"].astype('float')

    cash_price_eth = pd.read_csv("~/PycharmProjects/542project/ethereum_price.csv", parse_dates=['Date'])
    cash_price_eth.head()
    cash_price_eth['Volume'] = cash_price_eth["Volume"].astype('float')

    # run the prediction
    buildingset(cash_price_eth,"ethereum")
    buildingset(cash_price,"bitcoin")


if __name__ == "__main__":
    main()