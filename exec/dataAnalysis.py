import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp

def createAllCertainFormatFileList(filePath,fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
        if os.path.isfile(os.path.join(filePath, relativeFilename))
        if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList

def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame=rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame=cleanConditionDataFrame[cleanConditionDataFrame.beanEaten!=0]
    cleanbRealConditionDataFrame=cleanBeanEatenDataFrame.loc[cleanBeanEatenDataFrame['abnormalCondition'].isin(range(-5,6))]
    return cleanbRealConditionDataFrame

def calculateRealCondition(rawDataFrame):
    rawDataFrame['abnormalCondition']=(np.abs(rawDataFrame['bean2GridX'] - rawDataFrame['playerGridX'])+np.abs(rawDataFrame['bean2GridY'] - rawDataFrame['playerGridY']))-(np.abs(rawDataFrame['bean1GridX'] - rawDataFrame['playerGridX'])+np.abs(rawDataFrame['bean1GridY'] - rawDataFrame['playerGridY']))
    sheepDataFrameWithRealCondition=rawDataFrame.copy()
    return sheepDataFrameWithRealCondition


def readListCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for i in reader:
        a.append(json.loads(i[keyName]))
    f.close()
    return a


def readCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for i in reader:
        a.append(i[keyName])
    f.close()
    return a


if __name__=="__main__":
    # resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    # fileFormat = '.csv'
    # resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    # resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    # resultsDataFrame = pd.concat(resultsDataFrameList,sort=False)
    # resultsDataFrame=calculateRealCondition(resultsDataFrame)
    # resultsDataFrame=cleanDataFrame(resultsDataFrame)
    # participantsTypeList = ['Model' if 'Model' in name else 'Human' for name in resultsDataFrame['name']]
    # resultsDataFrame['participantsType']=participantsTypeList
    # resultsDataFrame['beanEaten']=resultsDataFrame['beanEaten']-1
    # trialNumberEatsheepDataFrame = resultsDataFrame.groupby(['name','abnormalCondition','participantsType']).sum()['beanEaten']
    # trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name','abnormalCondition','participantsType']).count()['beanEaten']
    # mergeConditionDataFrame = pd.DataFrame(trialNumberEatsheepDataFrame.values/trialNumberTotalEatDataFrame.values,index=trialNumberTotalEatDataFrame.index,columns=['eatsheepPercentage'])
    # mergeConditionDataFrame['eatOldPercentage']=1 - mergeConditionDataFrame['eatsheepPercentage']
    # mergeParticipantsDataFrame = mergeConditionDataFrame.groupby(['abnormalCondition','participantsType']).mean()
    # drawEatOldDataFrame=mergeParticipantsDataFrame['eatOldPercentage'].unstack('participantsType')
    # ax=drawEatOldDataFrame.plot.bar(color=['lightsalmon', 'lightseagreen'],ylim=[0.0,1.1],width=0.8)
    # pl.xticks(rotation=0)
    # ax.set_xlabel('Distance(sheep - old)',fontweight='bold')
    # ax.set_ylabel('Percentage of Eat Old',fontweight='bold')
    # plt.show()

    dirName = os.path.dirname(__file__)
    fileFolder = os.path.join(dirName, '..', 'results', 'testResult')
    csvList = []
    a = os.listdir(fileFolder)
    for j in a:
        if os.path.splitext(j)[1] == '.csv':
            csvList.append(fileFolder+'/'+j)

    sheepNumKey = 'sheepNums'
    sheepConcernKey = 'sheepConcern'
    blockSizeKey = 'blockSize'
    targetColorIndexesKey = 'targetColorIndexes'
    targetColorIndexesUnshuffledKey = 'targetColorIndexUnshuffled'
    sheepMaxSpeedKey = 'sheepMaxSpeed'
    trialScoreKey = 'trialScore'
    caughtTimesKey = 'caughtTimes'
    sheepEatenFlagKey = 'sheepEatenFlag'
    sheepCatchFlagKey = 'caughtFlag'
    
    sheepNum = []
    sheepConcern = []
    blockSize = []
    targetColorIndexes = []
    targetColorIndexesUnshuffled = []
    sheepMaxSpeed = []
    touchTimes = []
    caughtTimes = []
    trialScore = []
    caughtScoreProportion = []
    sheepEatenFlagEntropy = []
    sheepCatchFlagEntropy = []
    sheepEatenFlagPercentageSingletonFirst = []
    sheepCatchFlagPercentageSingletonFirst = []
    
    for i in range(len(csvList)):
        dfOri = pd.read_csv(csvList[i])
        readFun = lambda key: readListCSV(csvList[i], key)
        sheepNum.extend(readFun(sheepNumKey))
        blockSize.extend(readFun(blockSizeKey))
        sheepConcern.extend(readCSV(csvList[i], sheepConcernKey))
        targetColorIndexes.extend(readFun(targetColorIndexesKey))
        targetColorIndexesUnshuffled.extend(readFun(targetColorIndexesUnshuffledKey))
        sheepMaxSpeed.extend(readFun(sheepMaxSpeedKey))
        
        scoreRawData = readFun(trialScoreKey)
        caughtRawData  = readFun(caughtTimesKey)
        sheepEatenFlagRawData = readFun(sheepEatenFlagKey)
        sheepCatchFlagRawData = readFun(sheepCatchFlagKey)

        sheepEatenFlagPercentageSortBySingleton = []
        sheepCatchFlagPercentageSortBySingleton = []
        for trialIndex in range(len(scoreRawData)):
            colorIndex = targetColorIndexes[trialIndex]
            eatenFlag = sheepEatenFlagRawData[trialIndex]
            catchFlag = sheepCatchFlagRawData[trialIndex]
            # if np.sum(colorIndex) == 1:
                # singletonIndex = colorIndex.index(0)
                # eatenFlag.insert(0, eatenFlag.pop(singletonIndex))
                # catchFlag.insert(0, catchFlag.pop(singletonIndex))
            # if np.sum(colorIndex) == 2:
                # singletonIndex = colorIndex.index(1)
                # eatenFlag.insert(0, eatenFlag.pop(singletonIndex))
                # catchFlag.insert(0, catchFlag.pop(singletonIndex))
            sheepEatenFlagPercentageSortBySingleton.append(np.array(eatenFlag) / np.sum(eatenFlag))
            sheepCatchFlagPercentageSortBySingleton.append(np.array(eatenFlag) / np.sum(eatenFlag))
            
            
        touchRawData = (np.array(scoreRawData) - 3 * np.array(caughtRawData)) / 0.1
        caughtScorePropRawData = 3 * np.array(caughtRawData) / np.array(scoreRawData)
        sheepEatenFlagEntropyRawData = [sp.stats.entropy(sheepEatenFlag) for sheepEatenFlag in sheepEatenFlagRawData]
        sheepCatchFlagEntropyRawData = [sp.stats.entropy(sheepCatchFlag) for sheepCatchFlag in sheepCatchFlagRawData]
        
        trialScore.extend(scoreRawData)
        caughtTimes.extend(caughtRawData)
        touchTimes.extend(touchRawData)
        caughtScoreProportion.extend(caughtScorePropRawData)
        sheepEatenFlagEntropy.extend(sheepEatenFlagEntropyRawData)
        sheepCatchFlagEntropy.extend(sheepCatchFlagEntropyRawData)
        sheepEatenFlagPercentageSingletonFirst.extend(sheepEatenFlagPercentageSortBySingleton)
        sheepCatchFlagPercentageSingletonFirst.extend(sheepCatchFlagPercentageSortBySingleton)
        
    datas = {
        'sheepNum': sheepNum,
        'sheepConcern': sheepConcern,
        'blockSize': blockSize,
        'sheepMaxSpeed': sheepMaxSpeed, 
        'targetColorIndexes': [str(colorIndex) for colorIndex in targetColorIndexes],  
        'targetColorIndexesUnshuffled': [str(colorIndex) for colorIndex in targetColorIndexesUnshuffled],  
        'trialScore': trialScore,
        'caughtTimes': caughtTimes,
        'touchTimes': touchTimes,
        'caughtScoreProportion': caughtScoreProportion,
        'sheepEatenFlagEntropy': sheepEatenFlagEntropy,
        'sheepCatchFlagEntropy': sheepCatchFlagEntropy,
        'sheepEatenFlagPercentageSingletonFirst': sheepEatenFlagPercentageSingletonFirst,
        'sheepCatchFlagPercentageSingletonFirst': sheepCatchFlagPercentageSingletonFirst
    }

    dfTrialData = pd.DataFrame(datas)
    dfData = dfTrialData[(dfTrialData['blockSize'] == 0) & (dfTrialData['caughtTimes'] <= 550)]
    #dfData = dfTrialData[dfTrialData['caughtTimes'] <= 25]
    groupedData = dfData.groupby(['sheepConcern', 'sheepNum', 'blockSize', 'targetColorIndexesUnshuffled'])
    dfTotalScore = groupedData.sum()  # total score for every condition
    dfAverageScore = groupedData.mean()  # average score for every condition
    dfResetIndex = dfAverageScore.reset_index()
    dfSelfAverageScore = dfResetIndex[dfResetIndex.sheepConcern == 'self']
    print(dfSelfAverageScore)

    #sns.set_style("whitegrid")        # darkgrid(Default), whitegrid, dark, white, ticks
    f, ax = plt.subplots(figsize=(5, 5))
    # barplot: Default: np.mean
    # g = sns.barplot(x='targetColorIndexesUnshuffled', y='trialScore', data=dfData, estimator=np.mean, ci=95, capsize=.05, errwidth=2, color='Red', order = dfSelfAverageScore['targetColorIndexesUnshuffled'])#, palette='Reds')
    g = sns.barplot(y='sheepEatenFlagPercentageSingletonFirst', data=dfData, estimator=np.mean, ci=95, capsize=.05, errwidth=2, color='Red')#, order = dfSelfAverageScore['targetColorIndexesUnshuffled'])#, palette='Reds')
    # sns.barplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData, estimator=np.mean, ci=95, capsize=.05, errwidth=2, palette='Blues')
    # sns.boxplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData)

    #for index, row in dfSelfAverageScore.iterrows():
    #    g.text(index-3, row['trialScore']+2, round(row['trialScore'], 2), color="black", ha="center")

    #ax.set_ylim(0, 20)

    # 设置坐标轴下标的字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 设置坐标名字与字体大小
    plt.ylabel('average trialScore', fontsize=10)

    # 设置X轴的各列下标字体是水平的
    plt.xticks(rotation='horizontal')

    plt.show()
