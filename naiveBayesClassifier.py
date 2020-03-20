"""

-> Bayes Rule

Posterior Probability = Likelihood * Prior Knowledge / Evidence

P(H|E) = P(E|H) * P(H) / P(E)

H: Proposition
E: Evidence 
P(E|H): Likelihood Probability
P(H|E): Posterior Probability
P(E): Prior Probability of Evidence
P(H): Prior Probability of Proposition/Hypothesis

-> Bayes Rule in an ML Classifier perspective

P(Label|Features) = P(Features|Label) * P(Label) / P(Features)

Here P(Features|Label) gets tricky. 
Consider 4 features x1, x2, x3, x4. Label = 1

P(Features|Label)  = P(x1|x2, x3, x4, Label) * P(x1|x2, x3, x4, Label) * P(x2|x3, x4, Label) * P(x3|x4, Label) * P(x4|Label)

-> Here we make an assumption, that all features are independent to each other. This assumption is the "naive" in Naive Bayes

Thus, 

P(x1|x2, x3, x4, Label) = P(x1|Label) and so on.

P(Features|Label) = P(x1|Label) * P(x2|Label) * P(x3|Label) * P(x4|Label)
P(Features) = P(x1, x2, x3, x4) which is common for all. 


Finally, 

P(Label|Features) proportional to  P(x1|Label) * P(x2|Label) * P(x3|Label) * P(x4|Label) * P(Label)

-> Gaussian Naive Bayes

Likeihood calculation extends to continuous values as well.

Eg. 
Feature: Humidity 

| Humidity 	| Play 	|
|----------	|------	|
| 1.2      	| Yes  	|
| 0        	| No   	|
| 3.9      	| No   	|
| 1.4      	| Yes  	|
| 1.8      	| Yes  	|
| 4.1      	| No   	|

Now, we can't use the count of each category given a label to calculate the Likelihood. 
So the approach is, first assuming the probability density function is Normal Distribution
The,

yesMean = Mean of humidity values given label is Yes
yesVariance = Variance of humidity values given label is Yes

Likelihood(Humidity|Yes) =  exp(-1 * (yesVal - yesMean)**2 / (2 * yesVariance))  * 1 / sqrt(2 * pi * yesVariance) 
Similarly we may calculate Likelihood(Humidity|No).

"""
import math
import pprint
from random import shuffle


class NaiveBayesClassifier(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__featureNames = ["col_{}".format(i) for i in range(len(x[0]))]
        self.likelihood = {}

        self.priors = {}
        self.priorCount = {}

    def __calculatePriors(self):
        """
        Calculate the prior probabilities of dependent variable/output
        """

        for val in self.y:
            if self.priorCount.get(val):
                self.priorCount[val] += 1
            else:
                self.priorCount[val] = 1

        for key, val in self.priorCount.items():
            self.priors[key] = self.priorCount[key]/len(self.y)

    def __calculateLikelihood(self, feature):
        """
        Calculate likelihood of a feature
        Likelihood is a map of a feature
            * Keys - Categories in a feature
            * Values - Probabilities of each category in each class in Y

        {"Weather": {'Sunny': {'Play': .45, 'Don't Play: .55}, 'Rainy': {'Play': .69, 'Don't Play: .41}}}
        Feature -> Categories in the feature -> Categories assigned to each label
        """
        likelihood = {}
        for category in set(feature):
            temp = {}
            for val in set(self.y):
                temp[val] = 0
            likelihood[category] = temp

        for feat, out in zip(feature, self.y):
            likelihood[feat][out] += 1

        for probs in likelihood.values():
            for element in probs:
                probs[element] = probs[element] / self.priorCount[element]

        return likelihood

    def fit(self):
        """
        Training Naive Bayes means calculating the priors and likelihoods
        """
        self.__calculatePriors()
        for j in range(len(self.x[0])):
            feature = []
            for i in range(len(self.x)):
                feature.append(self.x[i][j])
            tmpLikelihood = self.__calculateLikelihood(feature)
            self.likelihood[self.__featureNames[j]] = tmpLikelihood

    def predict(self, x):
        """
        Calculate probability for each output class for the input using Bayes Formula
        Return the maximum probability class
        """
        probs = {val: 1 for val in set(self.y)}

        # Multiply the likelihood with each feature and each label
        idx = 0
        for val in x:
            for label in set(self.y):
                probs[label] *= self.likelihood[self.__featureNames[idx]][val][label]
            idx += 1

        # Multiply the prior probability
        for val in self.priors:
            probs[val] *= self.priors[val]

        # Normalize the probabilities
        total = sum(probs.values())

        for key in probs:
            probs[key] /= total

        print("\nClass Probabilities: ", probs)

        # Find maximum probability
        keyBest = max(probs, key=probs.get)
        return keyBest


class GaussianNaiveBayesClassifier(NaiveBayesClassifier):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.__normalizeX()

    def __calculateMean(self, x):
        """
        Calculate mean of values
        """
        meanVal = sum(x)/len(x)
        return meanVal

    def __calculateVariance(self, x, meanVal):
        """
        Calculate variance of values
        """
        varianceVal = sum([(xi - meanVal) ** 2 for xi in x]) / len(x)
        return varianceVal

    def __normalizeX(self):
        """
        Normalize the input data 
        """
        for j in range(len(self.x[0])):
            feat = []
            for i in range(len(self.x)):
                feat.append(self.x[i][j])
            temp = self.__normalize(feat)
            for k in range(len(self.x)):
                self.x[k][j] = temp[k]

    def __calculateGaussian(self, xi, xMean, xVar):
        """
        Calculate the Gaussian Func value
        Epsilon value is important here, to avoid zero variance scenerio
        """
        epsilon = math.pow(10, -1)
        firstTerm = 1/math.sqrt(2 * math.pi * (xVar + epsilon))
        secondTerm = math.exp(-0.5 * math.pow(xi - xMean, 2)/(xVar + epsilon))
        return firstTerm * secondTerm

    def __normalize(self, feature):
        """
        Normalize a feature 
        Helper in normalizing the entire input data
        """
        mean = self.__calculateMean(feature)
        var = self.__calculateVariance(feature, mean)
        normlizedFeature = [(x-mean)/var for x in feature]
        return normlizedFeature

    def __calculateLikelihood(self, feature):
        """
        Calculate likelihood of a feature, P(feature|output)
        Likelihood is a map of a feature 
            * Keys - Categories in a feature
            * Values - Probabilities of each category in each class in Y

        Here, we consider continuous feature values 

        Eg: 

        Feature - Humidity
        Output Labels - "Play" and "Don't Play" 

        likelihood = {'Play': {'Mean': .45, 'Variance': .29}, 'Don't Play: {'Mean': .69, 'Variance': .41}}
        Feature -> Categories in the feature -> Categories assigned to each label
        """
        likelihood = {}

        # Initialize the likelihood struct
        for category in set(self.y):
            temp = {"Mean": 0, "Variance": 0}
            likelihood[category] = temp

        # Get list of values corresponding to each category in outputs
        values = {}
        for category in set(self.y):
            values[category] = []

        for feat, out in zip(feature, self.y):
            values[out].append(feat)

        # Fore each category find mean and variance
        for category, val in values.items():

            mean = self.__calculateMean(val)
            var = self.__calculateVariance(val, mean)
            likelihood[category]["Mean"] = mean
            likelihood[category]["Variance"] = var

        return likelihood

    def fit(self):
        """
        Training Naive Bayes means calculating the priors and likelihoods
        """
        self._NaiveBayesClassifier__calculatePriors()
        print("Prior probability: ", self.priors)
        for j in range(len(self.x[0])):
            feature = []
            for i in range(len(self.x)):
                feature.append(self.x[i][j])
            tmpLikelihood = self.__calculateLikelihood(feature)
            self.likelihood[self._NaiveBayesClassifier__featureNames[j]
                            ] = tmpLikelihood

    def predict(self, x):
        """
        Calculate probability for each output class for the input using Bayes Formula
        Return the maximum probability class
        """

        # Initialize probs as prior probs
        probs = self.priors.copy()

        # Multiply the likelihood with each feature and each label
        # Each likelihood value is calculated from gaussian function
        idx = 0
        for val in x:
            for label in set(self.y):
                mean = self.likelihood[self._NaiveBayesClassifier__featureNames[idx]][label]["Mean"]
                var = self.likelihood[self._NaiveBayesClassifier__featureNames[idx]
                                      ][label]["Variance"]
                gaussVal = self.__calculateGaussian(val, mean, var)
                probs[label] *= gaussVal
            idx += 1

        # Normalize the probabilities
        total = sum(probs.values())

        for label in probs:
            probs[label] /= total

        # Find maximum probability
        keyBest = max(probs, key=probs.get)
        return keyBest

    def findAccuracy(self, yTrue, yPred):
        """
        Calculate accuracy 
        """
        return sum([1 if yPred[i] == yTrue[i] else 0 for i in range(len(yPred))])/len(yPred)


if __name__ == "__main__":

    dataX = []
    with open("golfData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(list(l.strip().split("\t")))

    with open("golfData/y.txt", "r") as f:
        dataY = f.read().splitlines()

    nbClassifier = NaiveBayesClassifier(dataX, dataY)
    nbClassifier.fit()

    print("\nPriors: ", nbClassifier.priors)
    print("\nLikelihood: ")
    pprint.pprint(nbClassifier.likelihood)

    test = ["Sunny", "Hot", "Normal", "False"]
    labels = list(set(dataY))

    print("\nResult: ", nbClassifier.predict(test))
