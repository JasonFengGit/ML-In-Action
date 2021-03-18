# coding: cp1252
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet): 
    # better version of setOfWords2Vec which stores the num 
    # of occurrenece instead of whether presence or absence
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # changed to + 1
    return returnVec

def trainNB0(trainMatrix, trainCategory): # train naive bayes classifier
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0]) # in vocab list
    pClass1 = sum(trainCategory) / numTrainDocs
    p0Num, p1Num = ones(numWords), ones(numWords) # avoid zeros
    p0Denom, p1Denom = 2, 2

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # occurrence / total num of words in the class
    # vector / scalar

    # log: avoid tiny numbers multiplied together and result in underflow
    # why it works: p1 * p2 * p3 -> log(p1) + log(p2) + log(p3) = log(p1 * p2 * p3)
    p1Vec = log(p1Num/p1Denom)  
    p0Vec = log(p0Num/p0Denom)
    return p0Vec, p1Vec, pClass1

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # pClass0 = 1 - pClass1 if there is only two classes
    # choose the class with greater probability
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pClass1 = trainNB0(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pClass1))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pClass1))
    
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString) # split by non-alphanumeric chars
    return [each.lower() for each in listOfTokens if len(each) > 2]

def spamTest(): # hold-out cross validation: randomly select a portion of the data 
    docList, classList, fullText = [], [], []
    for i in range(1, 26): # spams & hams 25 each
        # 1: spam 0: ham
        wordList = textParse(open('original/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('original/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #create vocabulary
    trainingSet = list(range(50))
    testSet=[]
    # randomly select 10 elements to become testSet
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
      
    trainMat, trainClasses = [], []
    for docIndex in trainingSet: #train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet: #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', errorCount/len(testSet), '\terror count:', errorCount)
    return errorCount/len(testSet)

if __name__ == "__main__":
    '''
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for post in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, post))
    p0V, p1V, pClass1 = trainNB0(trainMat, listClasses)
    print(p0V, p1V, pClass1)
    '''
    # testingNB()
    spamTest()
