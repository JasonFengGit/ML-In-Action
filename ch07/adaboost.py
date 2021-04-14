from numpy import *

def loadSimpData():
    dataMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == "lt":
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D): # D: weights
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = float("inf")

    for i in range(n): # loop through every feature
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ["lt", "gt"]:
                threshVal = (rangeMin + j * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    errorRate = 1
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print("D:", D.T)
        alpha = float(0.5*log((1-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("classEst:", classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon)) / D.sum()
        aggClassEst += alpha * classEst
        #print("aggClassest:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print("total error:", errorRate)
        if errorRate == 0:
            break
    print("train error rate:", errorRate)
    return weakClassArr, aggClassEst

def adaClassify(dataToClassify, classifierArr):
    dataMatrix = mat(dataToClassify)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return sign(aggClassEst)

def loadDataSet(filename):
    # assuming the last entry of each line is the label
    numFeat = len(open(filename).readline().split('\t'))
    dataMat, labelMat = [], []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) # cursor
    ySum = 0.0 # variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort() # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("AUC:", ySum * xStep)

if __name__ == "__main__":
    dataMat, classLabels = loadDataSet('original/horseColicTraining2.txt')
    testMat, testLabels = loadDataSet('original/horseColicTest2.txt')

    # the model gets overfitting when numClassifier is too large
    for numClassifier in [1, 10, 50, 100]:
        print('num classifiers:', numClassifier)
        classifierArr, aggClassEst = adaBoostTrainDS(dataMat, classLabels, numClassifier)
        prediction10 = adaClassify(testMat, classifierArr)
        errCount = (prediction10 != mat(testLabels).T).sum()
        print('test errorRate:', errCount/67)
        plotROC(aggClassEst.T, classLabels)
        print("-----------------------")