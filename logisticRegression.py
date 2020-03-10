"""
Logistic Regression
-------------------

-> A classification algorithm 
-> Linear Regression gives continuous output
-> Extends Linear Regression to discrete prediction values like, for eg. Spam or No Spam  

Data Flow 
---------

Data -> a Linear Model -> A logistic function like Sigmoid -> Output as probabilities (0-1) -> Compares with threshold -> Predict class

Change in Objective and Hypothesis
----------------------------------

-> In Linear Regression, our objective was to find the difference between the predicted and original value. 
   Minimizing that loss was the measure of success. 
   
    Hypothesis function: h(theta) = theta*x + intercept

-> In Logistic Regression, the outputs are probabilities, our objective is to maximize those probabilities. 

    Hypothesis function: h(theta) = sigmoid(theta*x + intercept)
    if theta*x + intercept >= 0, sigmoid(theta*x + intercept) = 1
    if theta*x + intercept < 0, sigmoid(theta*x + intercept) = 0

Cost Function 
-------------

-> Cost function of Linear Regression is MSE
-> In logistic regression, MSE will result in non-convex cost function since the objective has changed. 
-> Gradient Descent could always get stuck in a local minima, if the cost function is non-convex.

-> Cost Function = -1 * ( yTrue*log(yPred) + (1-yTrue)* log(1-yPred) )
-> yPred = sigmoid(theta*x + intercept)


-> This loss is called Binary Cross Entropy or Log Loss 

Why log loss?
-------------

Refer this link and you'll get to know about it clearly. 

-> https://github.com/KnetML/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section3-logistic-regression.ipynb
 

Optimization 
------------
-> Now, with new loss function, Gradient descent can be used to optimize Logistic Regression
-> For a logistic function f(x), the derivative will always be f(x)*(1-f(x))
-> This is a very useful relationship for finding the gradient of the loss function.

-> Surprisingly, the derivative will distill down to something similar to linear regression gradient.

partial derivative(loss function) w.r.t. theta = Sum([yPred - yTrue] * X)

"""

import random
import matplotlib.pyplot as plt
import math


class LogisticRegression():
    """
    Simple Logistic Regression with minimum library dependencies
    """

    def __init__(self):
        pass

    def _sigmoid(self, value):
        return 1/(1+math.exp(-1*value))

    def _init(self, rows, cols):
        """
        Initialize the learnable parameters
        """
        weights = [[random.random() for col in range(cols)] for row in range(rows)]
        
        bias = 0
        return weights, bias

    def _matrixMultiply(self, matrixA, matrixB):
        """
        Multiply two matrices
        Now we may appreciate the ready-to-go APIs from famous frameworks :D
        """
        if len(matrixA[0]) != len(matrixB):
            return None

        result = [[0 for col in range(len(matrixB[0]))] for row in range(len(matrixA))]
        # Iterate through rows of matrixA
        for rowA in range(len(matrixA)):
            # Iterate in columns of matrixB
            for colB in range(len(matrixB[0])):
                # Now iterate through each corresponding elements in that col
                for i in range(len(matrixB)):
                    result[rowA][colB] += matrixA[rowA][i] * matrixB[i][colB]
        return result

    def _matrixTranspose(self, matrix):
        """
        Given a matrix, return the transpose of the matrix
        """
        transpose = [[0 for col in range(len(matrix))]
                     for row in range(len(matrix[0]))]
        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                transpose[col][row] = matrix[row][col]
        return transpose

    def _calculatePredictions(self, weights, bias, data):
        """
        Calculate the predictions using formula Wx + b
        Wx is a matrix multiplication

        Weight matrix shape = number of features x 1
        Data matrix shape: number of data points x number of features 
        """

        result = 0
        tempResult = self._matrixMultiply(data, weights)
        result = [tempResult[i][0] + bias for i in range(len(tempResult))]
        result = [self._sigmoid(val) for val in result]
        return result

    def _calculateLossFunc(self, yPrediction, yTrue, weights, bias, regularizationCoeff=None):
        """
        Calculates the Log Loss with or without Ridge regularisation (L2 regularization)
        """
        # Calculate the difference between predicted and actual outputs

        loss = 0
        
        partOne = [yTrue[i] * math.log(yPrediction[i]) for i in range(len(yTrue))]
        partTwo = [(1-yTrue[i]) * math.log(1-yPrediction[i]) for i in range(len(yTrue))]

        loss = (-1/len(yTrue)) * \
            sum([partOne[i] + partTwo[i] for i in range(len(partOne))])

        regValue = 0

        # L2 or Ridge regularization added to cost function if required
        if regularizationCoeff:
            regValue = sum([val[0]**2 for val in weights]) * \
                regularizationCoeff / len(yTrue)

        loss = loss + regValue
        return loss

    def _getGradient(self, ypred, ytrue, x):
        """
        Calculates gradients of Log Error for weights and bias
        """
        # Learn theory to understand what is the derivative of MSE for slope and Y intercept
        diff = [[ypred[i] - ytrue[i]] for i in range(len(ytrue))]

        xTranspose = self._matrixTranspose(x)

        gradientWeightTemp = self._matrixMultiply(xTranspose, diff)

        gradientWeights = [[val[0]/len(diff)] for val in gradientWeightTemp]
        gradientBias = sum([x[0] for x in diff])/len(diff)

        return gradientWeights, gradientBias

    def _optimizer(self, x, yTrue, regularizationCoeff=None, learningRate=0.0001, epochs=100):
        """
        Performs the learning process
        """
        # Initialize the slope and intercept as Zeros
        weightHistory = []
        biasHistory = []

        myWeights, myBias = self._init(len(x[0]), 1)
        lossHistory = []

        # Iteratively update the coefficients, slope and y intercept
        for epoch in range(epochs):

            yPred = self._calculatePredictions(myWeights, myBias, x)
            # Calculate MSE loss between prediction and actual output
            loss = self._calculateLossFunc(yPred, yTrue, myWeights, myBias, regularizationCoeff)
            lossHistory.append(loss)

            if epoch % 10 == 0:
                print("Loss at {}th epoch: {}".format(epoch, loss))

            # Find the gradients 
            gradientWeights, gradientBias = self._getGradient(yPred, yTrue, x)

            # Find gradient for regularization part
            # Refer theory to understand the calculation
            regValueUpdate = 0
            if regularizationCoeff:
                regValueUpdate = sum([val[0] for val in myWeights]) * regularizationCoeff / len(x)

            # Gradient Descent update step
            myWeights = [[myWeights[i][0] - learningRate * gradientWeights[i][0]]
                       for i in range(len(myWeights))]
            if regularizationCoeff:
                myWeights = [[myWeights[i][0] - learningRate *
                           regValueUpdate] for i in range(len(myWeights))]

            myBias = myBias - learningRate * gradientBias

        print("Loss after {} epochs: {}".format(epochs, lossHistory[-1]))
        print("Training completed!")
        return myWeights, myBias, lossHistory

    def fit(self, x, yTrue, regularizationCoeff=None, learningRate=0.0001, epochs=600):
        """
        Trains the regressor with hyper parameters
        """
        return self._optimizer(x, yTrue, regularizationCoeff, learningRate, epochs)

    def predict(self, x, weights, bias):
        """
        Predict the result for a new test input
        """
        result = 0
        tempResult = self._matrixMultiply(x, weights)
        result = tempResult[0] + bias
        if self._sigmoid(result) >= 0.5:
            return 1
        return 0

    def calculateAccuracy(self, data, yTrue, weights, bias):
        """
        Calculate accuracy 
        """
        # Refer theory for understanding the calculation of R2
        yPred = self._calculatePredictions(weights, bias, data)
        yPred = [1 if val>=0.5 else 0 for val in yPred]
        temp = [1 if yPred[i] == yTrue[i] else 0 for i in range(len(yTrue))]
        accuracy = sum(temp)/len(yTrue)
        return accuracy

    def plotLoss(self, lossHistory, name = ""):
        """
        Plot the loss vs iterations graph
        """
        # Clear the canvas
        plt.clf()

        # Plot data 
        plt.plot(lossHistory, color="m")

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title("Loss vs Iterations Plot")

        if name:
            plt.savefig("plots/"+name)
            return
        plt.savefig("plots/lossGraph.png")


if __name__ == "__main__":

    #  Load dataset
    dataX = []
    with open("irisData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(list(map(float, l.strip().split("\t"))))
    with open("irisData/y.txt", "r") as f:
        dataY = list(map(float, f.readlines()))

    # Shuffle the dataset
    temp = list(zip(dataX, dataY))
    random.shuffle(temp)
    dataX, dataY = zip(*temp)
    dataX = list(dataX)
    dataY = list(dataY)

    # Split to training and validation set
    split = int(0.8 * len(dataX))

    trainX = dataX[:split]
    trainY = dataY[:split]

    testX = dataX[split:]
    testY = dataY[split:]

    logisticRegression = LogisticRegression()

    # Train the Linear Regressor and get optimal values for slope and Y intercept
    optWeights, optBias, lossHistory = logisticRegression.fit(
        trainX, trainY, regularizationCoeff=0.0, learningRate=0.01)

    # Calculate accuracy to evaluate model performance
    print("Training accuracy: ", logisticRegression.calculateAccuracy(
        trainX, trainY, optWeights, optBias))
    print("Testing accuracy: ", logisticRegression.calculateAccuracy(
        testX, testY, optWeights, optBias))
    logisticRegression.plotLoss(lossHistory)

