"""
Generally in Linear Regression, we assume the relationship (f) between y and x as a linear combination of x.

Simple Linear Regression 
------------------------
-> y = f(x), is represented as y = yIntercept + slope * x

Multiple Linear Regression 
--------------------------
-> y = f(x1, x2) is represented as y = yIntercept + slopeOne * x1 + slopeTwo * x2

Polynomial Regression
---------------------

-> y = f(x) is represented as y = yIntercept + slopeOne * x + slopeTwo * x^2 
-> y = f(x1, x2) represente as y = yIntercept + m1 * x1 + m2 * x2 + m3 * x1 * x2 + m4 * x1^2 + m5 * x2^2

"""

from random import shuffle

import matplotlib.pyplot as plt


class LinearRegression():
    """
    Simple Linear Regression with minimum library dependencies
    """

    def __init__(self):
        pass

    def __calculateLossFunc(self, yTrue, yPred, slope, regularizationCoeff=None):
        """
        Calculates the Mean Square Error with or without Ridge regularisation (L2 regularization)
        """
        # Calculate the difference predicted and actual outputs
        diff = [yTrue[i] - yPred[i] for i in range(len(yTrue))]
        # Square the differences
        squaredDiff = [val**2 for val in diff]
        # Take sum of the squares
        loss = sum(squaredDiff)/(2*len(yTrue))

        regValue = 0

        # L2 or Ridge regularization added to cost function if required
        if regularizationCoeff:
            regValue = (slope**2) * regularizationCoeff / 2*len(yTrue)

        loss = loss + regValue
        return loss

    def __getGradient(self, yPred, yTrue, x):
        """
        Calculates gradients of Mean Square Error for Slope and Y Intercept
        """

        # Learn theory to understand what is the derivative of MSE for slope and Y intercept
        diff = [yPred[i] - yTrue[i] for i in range(len(yTrue))]

        gradientSlopeTemp = [x[i]*diff[i] for i in range(len(diff))]
        gradientSlope = sum(gradientSlopeTemp)/len(diff)

        gradientYIntercept = sum(diff)/len(diff)

        return gradientSlope, gradientYIntercept

    def __optimizer(self, x, yTrue, regularizationCoeff=None, learningRate=0.0001, epochs=100):
        """
        Performs the learning process
        """
        # Initialize the slope and intercept as Zeros
        slope = [0]
        intercept = [0]
        loss = []

        # Iteratively update the coefficients, slope and y intercept
        for epoch in range(epochs):

            yPred = [slope[-1] * x[i] + intercept[-1] for i in range(len(x))]

            # Calculate MSE loss between prediction and actual output
            loss.append(self.__calculateLossFunc(
                yTrue, yPred, slope[-1], regularizationCoeff))

            if epoch % 10 == 0:
                print("Loss at {}th epoch: {}".format(epoch, loss[-1]))

            # Find the gradients
            gradientSlope, gradientYIntercept = self.__getGradient(
                yPred, yTrue, x)

            # Find gradient for regularization part
            # Refer theory to understand the calculation
            regValueUpdate = 0
            if regularizationCoeff:
                regValueUpdate = slope[-1] * regularizationCoeff / len(x)

            # Gradient Descent update step

            slope.append(slope[-1] - learningRate *
                         gradientSlope - learningRate * regValueUpdate)
            intercept.append(intercept[-1] - learningRate * gradientYIntercept)

        print("Loss after {} epochs: {}".format(epochs, loss[-1]))
        print("Slope after {} epochs: {}".format(epochs, slope[-1]))
        print("Y Intercept after {} epochs: {}".format(epochs, intercept[-1]))
        print("Training completed!")
        return slope, intercept, loss

    def fit(self, x, yTrue, regularizationCoeff=None, learningRate=0.0001, epochs=100):
        """
        Trains the regressor with hyper parameters
        """
        return self.__optimizer(x, yTrue, regularizationCoeff, learningRate, epochs)

    def plotRegressionLine(self, xVal, yVal, slope, intercept, name=None):
        """
        Plot the Regression Line on top of data
        """
        # Clear the canvas
        plt.clf()

        # Plot data
        plt.scatter(xVal, yVal, color="m", marker="o", s=30)

        # Calculate and plot the regression line
        yPred = [intercept + slope*val for val in xVal]
        plt.plot(xVal, yPred, color="g")

        plt.xlabel('x')
        plt.ylabel('y')

        if name:
            plt.savefig("plots/"+name)
            return
        plt.savefig("plots/regressionLine.png")

    def predict(self, x, slope, intercept):
        """
        Predict the result for a new test input
        """
        return [slope*val + intercept for val in x]

    def calculateR2(self, xVal, yVal, slope, intercept):
        """
        Determines how much of the total variation in Y (dependent variable) is 
        explained by the variation in X (independent variable)
        """

        # Refer theory for understanding the calculation of R2
        yPred = [intercept + slope*val for val in xVal]
        diff = [yVal[i]-yPred[i] for i in range(len(yVal))]
        squaredDiff = [val**2 for val in diff]
        numerator = sum(squaredDiff)

        yMean = sum(yVal)/len(yVal)
        diffMean = [yVal[i]-yMean for i in range(len(yVal))]
        squaredDiffMean = [val**2 for val in diffMean]
        denominator = sum(squaredDiffMean)

        return 1 - (numerator/denominator)


if __name__ == "__main__":

    #  Load dataset
    with open("weatherData/x.txt", "r") as f:
        dataX = list(map(float, f.readlines()))

    with open("weatherData/y.txt", "r") as f:
        dataY = list(map(float, f.readlines()))

    # Shuffle the dataset
    temp = list(zip(dataX, dataY))
    shuffle(temp)
    dataX, dataY = zip(*temp)

    # Split to training and validation set
    split = int(0.8 * len(dataX))

    trainX = dataX[:split]
    trainY = dataY[:split]

    testX = dataX[split:]
    testY = dataY[split:]

    linearRegression = LinearRegression()

    # Train the Linear Regressor and get optimal values for slope and Y intercept
    optSlope, optIntercept, loss = linearRegression.fit(
        trainX, trainY, regularizationCoeff=0.0, learningRate=0.0001)

    # Calculate R2 value to evaluate model performance
    print("Training R2: ", linearRegression.calculateR2(
        trainX, trainY, optSlope[-1], optIntercept[-1]))
    print("Testing R2: ", linearRegression.calculateR2(
        testX, testY, optSlope[-1], optIntercept[-1]))

    # Plot the regression line curves for training and test data
    linearRegression.plotRegressionLine(
        trainX, trainY, optSlope[-1], optIntercept[-1], name="trainingDataLinearRegression.png")
    linearRegression.plotRegressionLine(
        testX, testY, optSlope[-1], optIntercept[-1], name="testingDataLinearRegression.png")
