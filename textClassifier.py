import math
import random
import re

from naiveBayesClassifier import GaussianNaiveBayesClassifier

random.seed(0)


class TextClassifier():

    def __init__(self, docList, y):
        self.word2idx = {}
        self.termInverseFreqMap = {}

        self.__stopWords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                                'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
                                'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                                "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                                "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
        self.processedText = self.__cleanText(docList)
        self.trainY = y

        self.__buildVocabulary()
        self.__buildDocFreqMap()

    def __cleanText(self, textList):
        """
        Clean the input text
        1. Remove all special characters and numbers
        2. Remove all extra spaces
        3. Convert the text to lower case
        """
        processedText = []
        for text in textList:
            text = re.sub("[^a-zA-Z]+", " ", text)
            text = re.sub("\s+", " ", text)
            text = text.lower()
            processedText.append(text)
        return processedText

    def __checkVocabRequirements(self, word):
        """
        Check the criteria of adding a word to vocabulary
        """

        return word not in self.__stopWords and word not in self.word2idx and len(word) != 1

    def __buildVocabulary(self):
        """
        Build vocabulary for the input text 

        """
        i = 0
        for text in self.processedText:
            for word in text.split():
                if self.__checkVocabRequirements(word):
                    self.word2idx[word] = i
                    i += 1

    def __buildDocFreqMap(self):
        """
        Build a map with each word in vocabulary to how many documents contain that word
        """
        for doc in self.processedText:
            for word in set(doc.split()):
                if word in self.word2idx:
                    if self.termInverseFreqMap.get(word):
                        self.termInverseFreqMap[word] += 1
                    else:
                        self.termInverseFreqMap[word] = 1

    def buildTermFreqVector(self, text, norm=False):
        """
        Build a count vectorizer for the input documents
        For each word 

        """
        cleanedText = self.__cleanText([text])[0]
        cleanedTextList = cleanedText.split()
        termFreqVector = [0]*len(self.word2idx)

        for word in set(cleanedTextList):
            if word in self.word2idx.keys():
                termFreqVector[self.word2idx[word]
                               ] = cleanedTextList.count(word)

        if norm and sum(termFreqVector) != 0:
            termFreqVector = [val/sum(termFreqVector)
                              for val in termFreqVector]

        return termFreqVector

    def buildInvDocFrequencyMap(self, word):
        """
        Inverse Document Frequency  = log(No. of Docs / No. of docs in with the word is present) 
        """
        pass

    def buildTermFreqIDFVector(self, text, norm=False):
        """
        Calculate TFIDF vector 
        tfidf = termFreq * log(No. of Docs / No. of docs in with the word is present) 
        """
        cleanedData = self.__cleanText([text])[0]
        termFreqVector = self.buildTermFreqVector(cleanedData, norm)
        for word in set(cleanedData.split()):
            if word in self.word2idx.keys():
                docCount = self.termInverseFreqMap[word]
                termFreqVector[self.word2idx[word]
                               ] *= math.log(len(self.processedText)/(1+docCount))

        return termFreqVector

    def trainModel(self):
        trainX = []
        i = 0

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words=self.__stopWords)

        vectorizer.fit(self.processedText)

        self.word2idx = vectorizer.vocabulary_

        for text in self.processedText:
            tfIDFVector = self.buildTermFreqIDFVector(text)
            trainX.append(tfIDFVector)

        self.classifier = GaussianNaiveBayesClassifier(trainX, self.trainY)
        self.classifier.fit()

    def testModel(self, x):
        processedText = self.__cleanText([x])[0]
        testVector = self.buildTermFreqIDFVector(processedText)
        predictedLabel = self.classifier.predict(testVector)
        return predictedLabel

    def getAccuracy(self, yTrue, yPred):
        """
        Calculate accuracy 
        """
        return sum([1 if yPred[i] == yTrue[i] else 0 for i in range(len(yPred))])/len(yPred)


if __name__ == "__main__":

    dataX = []
    with open("spamData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(l.strip())
    dataY = []
    with open("spamData/y.txt", "r") as f:
        dataY = f.read().splitlines()

    # Shuffle the dataset
    temp = list(zip(dataX, dataY))
    random.shuffle(temp)
    dataX, dataY = zip(*temp)
    dataX = list(dataX)
    dataY = list(dataY)

    # Split to training and validation set
    dataX = dataX[:100]
    dataY = dataY[:100]

    split = int(0.8 * len(dataX))

    trainX = dataX[:split]
    trainY = dataY[:split]

    testX = dataX[split:]
    testY = dataY[split:]

    textClassifier = TextClassifier(trainX, trainY)

    textClassifier.trainModel()

    print("Prior Count: ", textClassifier.classifier.priorCount)

    yPredTrain = []
    for x in trainX:
        yPredTrain.append(textClassifier.testModel(x))

    yPredTest = []
    for x in testX:
        yPredTest.append(textClassifier.testModel(x))

    print("Training data accuracy: ",
          textClassifier.getAccuracy(trainY, yPredTrain))
    print("Testing data accuracy: ", textClassifier.getAccuracy(testY, yPredTest))
