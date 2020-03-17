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

"""

from random import shuffle
import pprint
class NaiveBayesClassifier(object):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.featureNames = ["col_{}".format(i) for i in range(len(x[0]))]
        self.likelihood = {}

        self.priors = {}
        self.priorCount = {}

    def calculatePriors(self):
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


    def calculateLikelihood(self, feature):
        """
        Calculate likelihood of a feature
        Likelihood is a map of a feature with keys as categories in a feature
        Keys are probabilities in each class in Y

        Eg. 
        Feature: Weather 
        Categories: Sunny, Rainy
        Y: Play, Don't Play 

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
        self.calculatePriors()
        for j in range(len(self.x[0])):
            feature = []
            for i in range(len(self.x)):
                feature.append(self.x[i][j])
            tmpLikelihood = self.calculateLikelihood(feature)
            self.likelihood[self.featureNames[j]] = tmpLikelihood

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
                probs[label] *= self.likelihood[self.featureNames[idx]][val][label]
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


if __name__ == "__main__":

    dataX = []
    with open("golfData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(list(l.strip().split("\t")))

    with open("golfData/y.txt", "r") as f:
        dataY = f.read().splitlines()

    featureNames = ["col_{}".format(i) for i in range(len(dataX[0]))]

    nbClassifier = NaiveBayesClassifier(dataX, dataY)
    nbClassifier.fit()

    print("\nPriors: ", nbClassifier.priors)
    print("\nLikelihood: ")
    pprint.pprint(nbClassifier.likelihood)

    test = ["Sunny", "Hot", "Normal", "False"]
    labels = list(set(dataY))

    print("\nResult: ", nbClassifier.predict(test))
