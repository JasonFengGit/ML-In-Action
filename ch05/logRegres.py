from numpy import *

def loadDataSet():
    dataMat, labelMat = [], []
    fr = open('original/testSet.txt', 'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    # return 1 / (1 + exp(-inX))
    # change to .5 * (1 + tanh(.5 * inX)) to avoid overflow
    return .5 * (1 + tanh(.5 * inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() # transform to column vector
    m, n = shape(dataMatrix)
    alpha = 0.001 # learning rate
    maxCycles = 500 # num of steps
    weights = ones((n, 1)) # initialize weights to ones, column vector
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # a estimate of the label [0, 1]
        error = labelMat - h # column vector
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    # one instance at a time
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + j + i) + 0.0001 # dynamic learning rate, decrease over time
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3, 3, 0.1)
    # w0 + w1*x + w2*y = 0(set *z* to zero to split two classes) -> y = (w0 - w1*x) / w2
    y = (-weights[0]-weights[1] * x) / weights[2] 
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('original/horseColicTraining.txt')
    frTest = open('original/horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]): # error
            errorCount += 1
    errorRate = errorCount / numTestVec
    print('error rate: ', errorRate, '\terror count:', errorCount)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print("after {} iterations the average error rate is: {}".format(numTests, errorSum/numTests))


if __name__ == "__main__":
    dataArr, labelMat = loadDataSet()
    '''
    weights = gradAscent(dataArr, labelMat)
    print(weights)
    plotBestFit(weights.getA())

    weights = stocGradAscent0(array(dataArr), labelMat)
    print(weights)
    plotBestFit(weights)
    
    weights = stocGradAscent1(array(dataArr), labelMat)
    plotBestFit(weights)
    '''
    multiTest()