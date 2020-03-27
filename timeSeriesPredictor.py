"""

Time series forecasting
=======================


Simple Moving Average
---------------------

Moving Average is a method used widely in time series data to predict future values. 

Select a window size first. Usually it can be 10, 20, 50, 100, 500 ... 

Simple Moving Average of i th element in the time series, SMA[i] = sum(All the previous elements of i upto window size)/ Window Size

Eg. 

Series = [1,2,3,4,5,6,7]
Window Size = 3

SMA[4] = sum(1+2+3)/3

Note: We can't find moving average until the length of series is at least equal to window size. Here we can't find moving average for 1, 2, 3.

Note: SMA can be used for smoothing the curve, that is creating an average curve which is less prone to rough differences in the original data.
Forecasting is predicting unknown values of future intervals. So use of SMA varies according to application. Here we discuss only the forecasting 
methods. This applies to following methods as well. 


Weighted Moving Average
-----------------------

Small difference from SMA. In SMA, we don't multiply anything with the input values. i.e. All the previous elements are equally treated.
What if we want to give more emphasis to later values than the earlier values? 

WMA[i] = sum(Values multiplied with weights)

Important note: The sum of the weighting should add up to 1 (or 100 percent).

Eg. 

Series = [1,2,3,4,5,6,7]
Window Size = 3
Weights = [1/6, 2/6, 3/6]

SMA[4] = sum(1x1/6 + 2x2/6 + 3x3/6)

Here we're giving more importance to values near to element in consideration than values in the distant past.

Exponential Moving Average
--------------------------

Bit more work here. 

Exponential Moving Average for tomorrow, EMA = Current value x Smoothing Factor + EMA of Today x (1 - Smoothing Factor)

Smoothing Factor = 2 / (Window Size + 1)

Note: For the first time calculation, we're not given with EMA, we may assume that EMA = SMA. 


What we are going to do with Moving Averages?
---------------------------------------------

Imagine I've people count in an airport for every month from 2019 to 2020. That is 12 months data. 

I want to predict the expected count of people for 2020 based on the previous year data. 
Now we can use moving averages to compute the prediction. Consider a three months window period

SMA
--- 

People count[Jan 2020] = Sum(People count[October 2019] + People count[November 2019] + People Count[December 2019])/3

WMA
---

People count[Jan 2020] = Sum(People count[October 2019] x 1/6 + People count[November 2019] x 2/6 + People Count[December 2019] x 3/6)

EMA
---

Smoothing factor = 2/(1+3) = 0.5
People count[Jan 2020] = 0.5 x People Count[December 2019] + (1 - 0.5)* EMA(December 2019)

"""

import math
import matplotlib.pyplot as plt


class TimeSeriesForecast(object):

    def __init__(self, x):
        self.x = x

    def __multiply(self, listA, listB):
        """
        Element wise multiplication of list
        """
        if len(listA) != len(listB):
            raise ValueError('List should be of same lengths')

        return [listA[i]*listB[i] for i in range(len(listA))]

    def rmsError(self, yTrue, yPred):
        """
        Calculate Mean Square Value loss
        """
        if len(yPred) != len(yTrue):
            raise ValueError("Lengths of predicted and actual values doesn't match.")

        noneCount = 0
        loss = 0
        for i in range(len(yTrue)):
            if yPred[i] == None:
                noneCount+=1
            else:
                loss += (yTrue[i] - yPred[i])**2
        loss = 0.5 * loss/len(yTrue)-noneCount
        return round(math.sqrt(loss), 2)

    def calcMovAvgs(self, windowLength):
        """
        Calculate simple moving average prediction in specified windowLength
        """
        movingAvgs = []
        cumSum = [0]
        n = len(self.x)

        for i in range(n):
            cumSum.append(cumSum[i] + self.x[i])
            if i >= windowLength:
                temp = (cumSum[i] - cumSum[i-windowLength])/windowLength
                movingAvgs.append(temp)
        return [None]*windowLength + movingAvgs

    def calcWeightedMovAvgs(self, windowLength, weights):
        """
        Calculate weighted moving average prediction
        """
        if sum(weights) != 1:
            raise ValueError('Weights should sum to 1')
        elif len(weights) != windowLength:
            raise ValueError("No. of weights should be equals to window size")

        weightedMovAvgs = []
        n = len(self.x)
        j = 0
        for i in range(n):
            if i >= windowLength:
                listA = self.x[j:j+windowLength]
                result = self.__multiply(listA, weights)
                weightedMovAvgs.append(sum(result))
                j += 1

        return [None]*windowLength + weightedMovAvgs

    def calcExpMovAvg(self, windowLength, smoothingFactor):
        """
        Calculate exponential moving average predicition
        """

        if smoothingFactor < 0 or smoothingFactor > 1:
            raise ValueError(
                "Value of smoothing factor should be in between 0-1")

        EMA_prev = sum(self.x[:windowLength])/windowLength
        n = len(self.x)

        EMA = []

        for i in range(n):
            if i >= windowLength:
                temp = smoothingFactor*self.x[i] + (1-smoothingFactor)*EMA_prev
                EMA.append(temp)
                EMA_prev = temp
        return [None]*windowLength + EMA

    def plotGraph(self, y1, y2, title="Y1 and Y2", xLabel="X", yLabel="Y", yOneLegend="Y1", yTwoLegend="Y2", name=None):
        """
        Plot the loss vs iterations graph
        """
        # Clear the canvas
        plt.clf()

        # Plot data
        plt.plot(y1, color="black", label=yOneLegend)
        plt.plot(y2, color="magenta", label=yTwoLegend)

        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.legend(loc="upper left")

        if name:
            plt.savefig("plots/"+name, bbox_inches="tight")
            return
        plt.savefig("plots/xyplot.png", bbox_inches="tight")


if __name__ == "__main__":
    dataX = []
    with open("airportData/x.txt", "r") as f:
        for l in f.readlines():
            temp = list(l.strip().split("\t"))
            temp[1] = int(temp[1])
            dataX.append(temp)

    peopleCount = [val[1] for val in dataX]

    timeSeriesPred = TimeSeriesForecast(peopleCount)

    movingAvg = timeSeriesPred.calcMovAvgs(3)
    weightedMovAvg = timeSeriesPred.calcWeightedMovAvgs(3, [1/6, 2/6, 3/6])
    expMovAvg = timeSeriesPred.calcExpMovAvg(3, 0.3)

    print("RMSE between data and moving average: ", timeSeriesPred.rmsError(peopleCount, movingAvg))
    print("RMSE between data and weighted moving average: ", timeSeriesPred.rmsError(peopleCount, weightedMovAvg))
    print("RMSE between data and exponential moving average: ", timeSeriesPred.rmsError(peopleCount, expMovAvg))

    timeSeriesPred.plotGraph(
        peopleCount, movingAvg, title="Data v/s Moving Avg", xLabel="Months", yLabel="PeopleCount", yOneLegend="Data", yTwoLegend="Mov. Avg", name="movAvgPlot.png")
    timeSeriesPred.plotGraph(
        peopleCount, weightedMovAvg, title="Data v/s Weighted Moving Avg", xLabel="Months", yLabel="PeopleCount", yOneLegend="Data", yTwoLegend="Wt. Mov. Avg", name="wtMovAvgPlot.png")
    timeSeriesPred.plotGraph(
        peopleCount, expMovAvg, title="Data v/s Exp. Moving Avg", xLabel="Months", yLabel="PeopleCount", yOneLegend="Data", yTwoLegend="Exp. Mov. Avg", name="expMovAvgPlot.png")
