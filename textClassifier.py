import re
import math

class TextClassifier():
    
    def __init__(self, docList):
        self.word2idx = {}
        self.idx2word = {}
        self.stopWords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
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
        self.processedText = self._cleanText(docList)
        self._buildVocabulary()
        self.termInverseFreqMap = {}
        self._buildDocFreqMap()

    def _cleanText(self, textList):
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

    def _buildVocabulary(self):
        """
        Build vocabulary for the input text 

        """
        i = 0
        for text in self.processedText:
            for word in text.split():
                if word not in self.stopWords and word not in self.word2idx and word not in self.idx2word:
                    self.word2idx[word] = i
                    self.idx2word[i] = word
                    i+=1

    def _buildDocFreqMap(self):
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
        
    def buildTermFreqVector(self, text, norm=True):
        """
        Build a count vectorizer for the input documents
        For each word 

        """
        processedText = self._cleanText([text])[0]
        termFreqVector = [0]*len(self.word2idx)
        for word in processedText.split():
            if word in self.word2idx.keys():
                termFreqVector[self.word2idx[word]] = processedText.count(word)
        if norm and sum(termFreqVector) != 0:
            termFreqVector = [val/sum(termFreqVector) for val in termFreqVector]
        return termFreqVector

    def buildTermFreqIDFVector(self, text, norm=True):
        """
        Calculate TFIDF vector 
        tfidf = termFreq * log(No. of Docs / No. of docs in with the word is present) 
        """
        processedText = self._cleanText([text])[0]
        termFreqVector = self.buildTermFreqVector(text, norm)
        for word in processedText.split():
            try:
                docCount = self.termInverseFreqMap[word]
                termFreqVector[self.word2idx[word]] = math.log(len(termFreqVector)/docCount)
            except KeyError:
                docCount = 1
            
        return termFreqVector

    def trainModel(self, parameter_list):
        raise NotImplementedError

    def testModel(self, parameter_list):
        raise NotImplementedError
    
    def getAccuracy(self, parameter_list):
        raise NotImplementedError


if __name__ == "__main__":

    dataX = []
    with open("spamData/x.txt", "r") as f:
        for l in f.readlines():
            dataX.append(l.strip())

    dataX = dataX[:100]

    with open("spamData/y.txt", "r") as f:
        dataY = f.read().splitlines()

    textClassifier = TextClassifier(dataX)

    print(textClassifier.buildTermFreqVector("TELL HER I SAID EAT SHIT."))
    print(textClassifier.buildTermFreqIDFVector("TELL HER I SAID EAT SHIT."))


