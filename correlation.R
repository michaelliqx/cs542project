library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
data<-read.csv("../input/ethereum_dataset.csv")
data$Date<-as.Date(data$Date,format="%Y/%M/%D")
a<-log(1+data$eth_etherprice)
library(zoo)
library(xts)
#week data
a.ts<-xts(a,as.Date(1:934,origin="2015-06-01"))
week_data<-apply.weekly(a.ts,mean)

library(astsa)
sarima(week_data,10,1,5)
acf2((week_data))
forecast<-sarima.for(week_data,20,1,2,1,1,2,1,9)
week2_data<-week_data[1:134]
sarima(week2_data,7,1,2)
acf2((week2_data))
forecast<-sarima.for(week2_data,20,7,1,2)
# Input data files are available in the "../input/" directory.
system("ls ../input")
# The final graph is the forecasting for the confidence interval of 95% and 80%
