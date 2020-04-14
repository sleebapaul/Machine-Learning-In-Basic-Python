"""
Check if the GaussianNB classifier works for continuous data 

Data used: Iris Dataset
"""

import random

from naiveBayesClassifier import GaussianNaiveBayesClassifier

# Load Dataset

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

# Modelling and training
clf = GaussianNaiveBayesClassifier(trainX, trainY)
clf.fit()

# The prior and likelihood probabilities
print("Prior Count: ", clf.priorCount)
print("Likelihood: ", clf.likelihood)

# Testing in testing and training data
yPredTrain = []
for x in trainX:
    yPredTrain.append(clf.predict(x))
print("Training data accuracy: ", clf.findAccuracy(trainY, yPredTrain))

yPredTest = []
for x in testX:
    yPredTest.append(clf.predict(x))
print("Testing data accuracy: ", clf.findAccuracy(testY, yPredTest))
