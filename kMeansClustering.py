"""
K-Means Clustering Algorithm
----------------------------

-> It is an unsupervised algorithm. We don't need a labelled dataset.
-> K-Means cluster the input data into a given number of groups. 

Steps
-----

1. Each group is assigned with a centroid / centre point. 
2. Based on the minimum distance between the a data point and the centroids, the group label is assigned to that point. 
3. Once each point in the dataset is assigned to a group, an iteration will be completed.
4. Now the centroids are updated by averaging the points in a group.
5. Goto step 2, with the updated centroids. Continue untils a maximum iterations count or centroid don't change above a limit. 

"""

from random import Random, sample, shuffle
from math import sqrt

import matplotlib.pyplot as plt


class KMeansCluster(object):
    
    def __init__(self, numClusters, maxIterations=200, randomState=123):
        self.numClusters = numClusters
        self.maxIterations = maxIterations
        self.randomSeed = Random(randomState)
        self.centroids = []
        self.clusterMap = {i:[] for i in range(numClusters)}

    def __calculateMean(self, feature):
        """
        Calculate mean of a feature
        """
        meanValue = sum(feature)/len(feature)
        return meanValue

    def __calculateSD(self, feature, meanValue):
        """
        Calculate standard deviation of a feature
        """
        varianceValue = sum([(xi - meanValue) ** 2 for xi in feature]) / len(feature)
        return sqrt(varianceValue)

    def __normalizeFeature(self, feature):
        """
        Normalize a feature 
        Helper in normalizing the entire input data
        """
        mean = self.__calculateMean(feature)
        var = self.__calculateSD(feature, mean)
        normlizedFeature = [(x-mean)/var for x in feature]
        return normlizedFeature

    def __normalizeData(self, data):
        """
        Normalize the entire input data 
        """
        for j in range(len(data[0])):
            
            feature = []
            
            for i in range(len(data)):
                feature.append(data[i][j])

            temp = self.__normalizeFeature(feature)
            for k in range(len(data)):
                data[k][j] = temp[k]
        return data
    
    def __initCentroids(self, data):
        """
        Initialize the centroids as random values from input data.
        """
        self.centroids = sample(data, self.numClusters)

    def __calculateDistance(self, pointA, pointB):
        """
        Calculate Euclidean distance between pointA and pointB
        """
        assert len(pointA) == len(pointB), "Size of data points are not same."
        dist = 0
        for i in range(len(pointA)):
            dist += (pointA[i] - pointB[i])**2
        return sqrt(dist)

    def __updateCentroids(self):
        """
        Update centroids after an iteration
        New centroid = sum of values of that assigned to that group/No. of such values
        """
        for i in range(len(self.centroids)):
            centroid = self.centroids[i]
            correspondingDataPoints = self.clusterMap[i]
            tmp = [0]*len(centroid)
            for dataPoint in correspondingDataPoints:
                for j in range(len(tmp)):
                    tmp[j] += dataPoint[j]
            tmp = [tmp[k]/len(correspondingDataPoints) for k in range(len(tmp))]
            self.centroids[i] = tmp

    def fit(self, data):
        """
        Train the K-Means on the data
        """
        shuffle(data)
        # data = self.__normalizeData(data)
        self.__initCentroids(data)

        step = 0
        while step<self.maxIterations:
            for value in data:
                temp = []
                for centroid in self.centroids:
                    loss = self.__calculateDistance(value, centroid)
                    temp.append(loss)
                minLossGroup = temp.index(min(temp))
                self.clusterMap[minLossGroup].append(value)
            
            self.__updateCentroids()
            self.clusterMap = {i:[] for i in range(self.numClusters)}
            step+=1
        
        for value in data:
            temp = []
            for centroid in self.centroids:
                loss = self.__calculateDistance(value, centroid)
                temp.append(loss)
            minLossGroup = temp.index(min(temp))
            self.clusterMap[minLossGroup].append(value)
        
    def predict(self, dataPoint):
        """
        Predict the cluster of a given data point
        """
        minVal = float("inf")
        groupID = None

        for i, centroid in enumerate(self.centroids):
            loss = self.__calculateDistance(dataPoint, centroid)
            if loss < minVal:
                minVal = loss
                groupID = i
        return groupID+1

def plotGroups(clusterMap, name=""):
    """
    Plot the clusters
    """
    # Clear the canvas
    plt.clf()

    # Plot data
    group = clusterMap[0]
    groupX = []
    groupY = []
    for val in group:
        groupX.append(val[0])
        groupY.append(val[1])
    plt.scatter(groupX, groupY, color="black", label="Group 1")
        
    group = clusterMap[1]
    groupX = []
    groupY = []
    for val in group:
        groupX.append(val[0])
        groupY.append(val[1])
    plt.scatter(groupX, groupY, color="magenta", label="Group 2")
    plt.legend(loc="upper left")

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Scatter Plot among features")

    if name:
        plt.savefig("plots/"+name)
        return
    plt.savefig("plots/clusterGraph.png")

def plotWithOriginalLabels():
    import pandas as pd
    data = pd.read_csv("clusterData/x.txt", sep="\t", header=None)
    y = pd.read_csv("clusterData/y.txt", delimiter="\n", header=None)
    plt.clf()
    plt.title("Feature X1 vs X2")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(data[0], data[1], c=y[0])
    plt.savefig("plots/clusterGraphFromLabels.png")


if __name__ == "__main__":
    
    # Load dataset
    dataX = []
    with open("clusterData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(list(map(float, l.strip().split("\t"))))
        with open("clusterData/y.txt", "r") as f:
            dataY = list(map(float, f.readlines()))


    clusterObj = KMeansCluster(2)
    clusterObj.fit(dataX)
    plotWithOriginalLabels()
    plotGroups(clusterObj.clusterMap)
    print("Predicted group of [4.5, 3.2] is: ", clusterObj.predict([4.5, 3.2]))

    


    







    









    
