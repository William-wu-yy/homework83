from math import log
import operator



def createDataSet():
    dataSet = [
        [1, 1, 0, 0, '+'],
        [0, 1, 0, 0, '+'],
        [1, 1, 1, 1, '-'],
        [1, 1, 0, 0, '-'],
        [1, 0, 1, 0, '+'],
        [0, 0, 0, 1, '+'],
        [1, 1, 1, 0, '-'],
        [0, 1, 1, 1, '-']
    ]
    labels = ['X1', 'X2', 'X3', 'X4']
    return dataSet, labels



def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print(f"特征 X{i + 1} 的信息增益为: {infoGain:.3f}")
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    print(f"选择的最优特征是：X{bestFeature + 1}")
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]


    if classList.count(classList[0]) == len(classList):
        print(f"所有类相同，直接返回类标签：{classList[0]}")
        return classList[0]


    if len(dataSet[0]) == 1:
        return majorityCnt(classList)


    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    print(f"当前划分特征: {bestFeatLabel}")
    myTree = {bestFeatLabel: {}}

    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        print(f"划分特征 {bestFeatLabel} = {value} 时，子数据集为：")
        for subData in subDataSet:
            print(subData)
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)

    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print("开始构建决策树...\n")
    myTree = createTree(dataSet, labels)
    print("\n构建的决策树为: ")
    print(myTree)

import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="round,pad=0.5", fc="0.8", ec="k", lw=1.5)
leafNode = dict(boxstyle="round4,pad=0.5", fc="0.9", ec="k", lw=1.5)
arrow_args = dict(arrowstyle="<|-", lw=1.5, color='gray')



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)



def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs



def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth



def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    plt.text(xMid, yMid, txtString, va="center", ha="center", color='blue', fontsize=10)



def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1, facecolor='white', figsize=(10, 8))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()



if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print("开始构建决策树...\n")
    myTree = createTree(dataSet, labels)
    print("\n构建的决策树为: ")
    print(myTree)

    createPlot(myTree)
