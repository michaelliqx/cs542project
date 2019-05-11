#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import plotly as py
import plotly.graph_objs as go
import os


# print(os.listdir("./dataset"))

def main():
    csv = os.listdir("./dataset")
    # file list
    csvList = [x[:-4] for x in csv]
    dataList = csvList.copy()
    for i in range(0, len(csv)):
        temp = "./dataset/" + csv[i]
        csvList[i] = pd.read_csv(temp)
        print("csvList info: ", csvList[i].info)

    priceList = []
    # keep the style of Cryptocurrency and add a column "Name"
    for i in range(0, len(csvList)):
        if dataList[i].count("price") > 0:
            csvList[i]["Name"] = dataList[i][:-6]
            print(csvList[i]["Name"])
            # change Market Cap to Market_Cap
            csvList[i].columns = [col.replace(' ', '_') for col in csvList[i].columns]            
            priceList.append(csvList[i])

    print(priceList[0].columns)

    # change Date format
    for i in range(0, len(priceList)):
        priceList[i].Date = pd.to_datetime(priceList[i].Date, infer_datetime_format=True)

   
    # get a list with all kinds of Cryptocurrency
    comparePrice = pd.concat(priceList, ignore_index=True)

    print(comparePrice.head())
    print(comparePrice.info())
    # convert the data type of Market Cap and Volume prom string to int
    # drop those data which only has "-"
    procData = comparePrice.copy()
    procData = procData[procData.Market_Cap != "-"]
    procData.Market_Cap = procData.Market_Cap.str.replace(",", "").astype(int)
    procData = procData[procData.Volume != "-"]
    procData.Volume = procData.Volume.str.replace(",", "").astype(int)
    print(procData.head())
    print(procData.info())
    print(procData.High[10])

    pd.options.display.float_format = '{:,.2f}'.format
    getMean = procData.groupby("Name").mean()
    print(getMean)



    trace = go.Pie(labels=getMean.index, values=getMean.Market_Cap)
    data = [trace]
    layout = dict(title=str("Market Cap "))
    fig = dict(data=data, layout=layout)
    py.offline.plot(fig)
    
    # plot the best 5 Cryptocurrency's market cap
    sortMeanList = getMean.sort_values(by=['Market_Cap'],ascending=False).head(5).index 
    
    tempGroup = procData.groupby('Name')
    data = []

    for currency in sortMeanList[::-1]:
        curGroup = tempGroup.get_group(currency)
        getDate = curGroup['Date'].tolist()
        marketCap = curGroup['Market_Cap'].tolist()
        zeros = [0] * len(getDate)
        
        # x for date, y for currency type, z for market cap
        data.append(dict(
            type='scatter3d',
            mode='lines',
            x=getDate + getDate[::-1] + [getDate[0]],  
            y=[currency] * len(getDate),
            z=marketCap + zeros + [marketCap[0]],
            name=currency,
            line=dict(
                width=4
            ),
        ))

    layout = dict(
        title='Top 5 Cryptocurrencies Market Cap',
        scene=dict(
            xaxis=dict(title='Dates'),
            yaxis=dict(title='Cryptocurrencies Type'),
            zaxis=dict(title='Market Cap'),
            camera=dict(
                eye=dict(x=-1.7, y=-1.7, z=0.5)
            )
        )
    )

    fig = dict(data=data, layout=layout)
    py.offline.plot(fig)
    

    
    


if __name__ == '__main__':
    main()
