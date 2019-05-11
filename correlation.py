import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
#from matplotlib.finance import candlestick_ohlc
#configuring the Environment
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

crypto_data = {}
crypto_data['bitcoindataset'] = pd.read_csv('../input/bitcoin_dataset.csv')
crypto_data['ethereumdataset'] = pd.read_csv("../input/ethereum_dataset.csv")

temp_df = crypto_data['bitcoindataset']
corrmat = temp_df.corr(method='spearman')
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Correlation Map for Bitcoin", fontsize=12)
plt.show()
temp_df.corr(method='spearman')

temp_df = crypto_data['ethereumdataset']
corrmat = temp_df.corr(method='spearman')
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Correlation Map for Ethereum", fontsize=12)
plt.show()
temp_df.corr(method='spearman')
