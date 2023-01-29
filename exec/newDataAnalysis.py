import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import random
import copy

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
    
def splitListColom(dataDf, columns):
    # columns = list(dataDf.columns)
    for c in columns:
        sheepCol = dataDf.pop(c)
        print(c)
        maxLen = max(list(map(lambda x:len(x) if (isinstance(x, list) or isinstance(x, np.ndarray)) else 1, sheepCol.values))) # list max len
        sheepCol = sheepCol.apply(lambda x: list(x)+[np.nan]*(maxLen - len(x)) if (isinstance(x, list) or isinstance(x, np.ndarray)) else [x]+[np.nan]*(maxLen - 1)) 
        sheepCol = np.array(sheepCol.tolist()).T 
        # print(sheepCol)
        for i, j in enumerate(sheepCol):
            dataDf[c + str(i)] = j
    return dataDf
     


if __name__=="__main__":

    dirName = os.path.dirname(__file__)
    fileFolder = os.path.join(dirName, '..', 'results', 'Expt 1 Data')
    fileFolder = os.path.join(dirName, '..', 'results', 'testResult')
    csvList = []
    a = os.listdir(fileFolder)
    for j in a:
        if os.path.splitext(j)[1] == '.csv':
            csvList.append(os.path.join(fileFolder, j))

    targetColorIndexesKey = 'targetColorIndexes'
    targetColorIndexesUnshuffledKey = 'targetColorIndexUnshuffled'
    sheepEatenFlagKey = 'sheepEatenFlag'
    sheepCatchFlagKey = 'caughtFlag'
    
    allDf = []
    
    for i in range(len(csvList)):
        dfOriginal = pd.read_csv(csvList[i])
        readFun = lambda key: readListCSV(csvList[i], key)
        targetColorIndexes = readFun(targetColorIndexesKey)
        targetColorIndexUnshuffled = readFun(targetColorIndexesUnshuffledKey)
        sheepEatenFlagRawData = readFun(sheepEatenFlagKey)
        sheepCatchFlagRawData = readFun(sheepCatchFlagKey)
        
        dfOriginal['targetColorIndexes'] = targetColorIndexes
        dfOriginal['targetColorIndexUnshuffled'] = targetColorIndexUnshuffled
        dfOriginal['sheepNum'] = [len(colorIndex) for colorIndex in targetColorIndexUnshuffled]
        dfOriginal['singletonOrNot'] = [ ((np.sum(colorIndex) == 1) or (np.sum(colorIndex) == 4))   for colorIndex in targetColorIndexUnshuffled]
        print(dfOriginal['singletonOrNot'] )
        dfOriginal['sheepEatenFlag'] = sheepEatenFlagRawData
        dfOriginal['caughtFlag'] = sheepCatchFlagRawData
        
        sheepEatenFlagSortBySingleton = []
        sheepCatchFlagSortBySingleton = []
        sheepEatenFlagPercentageSortBySingleton = []
        sheepCatchFlagPercentageSortBySingleton = []
        isSingletonEatenMost = []
        isSingletonCatchMost = []
        for trialIndex in range(len(targetColorIndexes)):
            colorIndex = targetColorIndexes[trialIndex]
            eatenFlag = sheepEatenFlagRawData[trialIndex]
            catchFlag = sheepCatchFlagRawData[trialIndex]
            # if np.sum(colorIndex) == 1 and len(colorIndex) == 3:
            # if colorIndex == [0, 0, 0, 0 ,1]:
                # singletonIndex = colorIndex.index(1)
                # isSingletonEatenMostBool = int(eatenFlag.index(max(eatenFlag)) == singletonIndex)
                # isSingletonCatchMostBool = int(catchFlag.index(max(catchFlag)) == singletonIndex)
                # eatenFlag.insert(0, eatenFlag.pop(singletonIndex))
                # catchFlag.insert(0, catchFlag.pop(singletonIndex))
            # # if np.sum(colorIndex) == 2 and len(colorIndex) == 3:
            # if colorIndex == [0, 1, 1, 1 ,1]:
                # singletonIndex = colorIndex.index(0)
                # isSingletonEatenMostBool = int(eatenFlag.index(max(eatenFlag)) == singletonIndex)
                # isSingletonCatchMostBool = int(catchFlag.index(max(catchFlag)) == singletonIndex)
                # eatenFlag.insert(0, eatenFlag.pop(singletonIndex))
                # catchFlag.insert(0, catchFlag.pop(singletonIndex))

            # if not (colorIndex == [0, 0, 0, 0 ,1] or colorIndex == [0, 1, 1, 1 ,1]):
                # isSingletonEatenMostBool = np.nan
                # isSingletonCatchMostBool = np.nan
                # random.shuffle(eatenFlag)
                # random.shuffle(catchFlag)

            # sheepEatenFlagSortBySingleton.append(eatenFlag)
            # sheepCatchFlagSortBySingleton.append(catchFlag)

            # if np.sum(eatenFlag) == 0 or (not (colorIndex == [0, 0, 0, 0 ,1] or colorIndex == [0, 1, 1, 1 ,1])):
                # isSingletonEatenMostBool = np.nan
                # sheepEatenFlagPercentage = np.array([np.nan] * len(eatenFlag))
            # else:
                # sheepEatenFlagPercentage = np.array(eatenFlag) / np.sum(eatenFlag)
            # isSingletonEatenMost.append(isSingletonEatenMostBool)
            # sheepEatenFlagPercentageSortBySingleton.append(sheepEatenFlagPercentage)
            # if np.sum(catchFlag) == 0 or (not (colorIndex == [0, 0, 0, 0 ,1] or colorIndex == [0, 1, 1, 1 ,1])):
                # isSingletonCatchMostBool = np.nan
                # sheepCatchFlagPercentage = np.array([np.nan] * len(catchFlag))
            # else:
                # sheepCatchFlagPercentage = np.array(catchFlag) / np.sum(catchFlag)
            # isSingletonCatchMost.append(isSingletonCatchMostBool)
            # sheepCatchFlagPercentageSortBySingleton.append(sheepCatchFlagPercentage)
            
        # dfOriginal['isSingletonEatenMost'] = isSingletonEatenMost
        # dfOriginal['isSingletonCatchMost'] = isSingletonCatchMost
        # dfOriginal['sheepEatenFlagSortBySingleton'] = sheepEatenFlagSortBySingleton
        # dfOriginal['sheepCatchFlagSortBySingleton'] = sheepCatchFlagSortBySingleton
        # dfOriginal['sheepEatenFlagPercentageSortBySingleton'] = sheepEatenFlagPercentageSortBySingleton
        # dfOriginal['sheepCatchFlagPercentageSortBySingleton'] = sheepCatchFlagPercentageSortBySingleton

        scoreRawData = dfOriginal['trialScore'] .values
        caughtRawData  = dfOriginal['caughtTimes'] .values
        
        touchRawData = (np.array(scoreRawData) - 3 * np.array(caughtRawData)) / 0.1
        caughtScorePropRawData = 3 * np.array(caughtRawData) / np.array(scoreRawData)
        sheepEatenFlagEntropyRawData = [sp.stats.entropy(sheepEatenFlag) for sheepEatenFlag in sheepEatenFlagRawData]
        sheepCatchFlagEntropyRawData = [sp.stats.entropy(sheepCatchFlag) for sheepCatchFlag in sheepCatchFlagRawData]
        dfOriginal['touchTimes'] = touchRawData
        dfOriginal['caughtScoreProportion'] = caughtScorePropRawData
        dfOriginal['sheepEatenFlagEntropy'] = sheepEatenFlagEntropyRawData
        dfOriginal['sheepCatchFlagEntropy'] = sheepCatchFlagEntropyRawData
        
        #string condition label to groupby
        dfOriginal['targetColorIndexes']  = [str(e) for e in targetColorIndexes]
        dfOriginal['targetColorIndexUnshuffled']  = [str(e) for e in targetColorIndexUnshuffled]

        allDf.append(dfOriginal)
    
    wolfType = 'Human'
    measureLabelDict = {'trialScore': 'Accumulated Reward', 'caughtTimes': 'Number of Catches'}
    measure = 'trialScore'
    measureLabel = measureLabelDict[measure]
    
    dfTrialData = pd.concat(allDf)
    dfTrialData= splitListColom(dfTrialData, ['sheepEatenFlag', 'caughtFlag'])#, 'sheepEatenFlagSortBySingleton', 'sheepCatchFlagSortBySingleton', 'sheepEatenFlagPercentageSortBySingleton', 'sheepCatchFlagPercentageSortBySingleton'])
    # dfTrialData= splitListColom(dfTrialData, ['sheepEatenFlag', 'caughtFlag', 'sheepEatenFlagSortBySingleton', 'sheepCatchFlagSortBySingleton', 'sheepEatenFlagPercentageSortBySingleton', 'sheepCatchFlagPercentageSortBySingleton'])
    dfData = copy.deepcopy(dfTrialData[(dfTrialData['sheepNum'] <= 4) & (dfTrialData['caughtTimes'] <= 1000)])
    
    dfData['Stag Speed Multiplier'] = dfData['sheepMaxSpeed']
    dfData['Set Size'] = dfData['sheepNum']
    
    groupedData = dfData.groupby(['name', 'Set Size', 'Stag Speed Multiplier'])#, 'blockSize', 'targetColorIndexUnshuffled', 'singletonOrNot'])
    dfOutputStat = groupedData['trialScore'].agg(['mean', 'std'])
    dfOutputStat['SE'] = dfOutputStat['std']/np.sqrt(len(allDf))
    dfToStat = dfOutputStat.reset_index()
    dfToStat[measureLabel] = measureLabel
    print(dfToStat)
    dfOutputStat.to_csv(os.path.join(dirName, '..', 'results', 'shareRewardBaseResult_wolfFlex_sheep6_No5Folder_120000eps_ToStat', wolfType + '_' + measureLabel +'.csv'))
    
    dfMeanBySubAndCondtition = groupedData.mean()  # average score for every condition
    dfMeanBySubAndCondtitionResetIndex = dfMeanBySubAndCondtition.reset_index()
    dfGroupByCondtitionBaseSub = dfMeanBySubAndCondtition.groupby(['Set Size', 'Stag Speed Multiplier'])
    dfOutputPlot = dfGroupByCondtitionBaseSub['trialScore'].agg(['mean', 'std'])
    dfOutputPlot['SE'] = dfOutputPlot['std']/np.sqrt(len(allDf))
    dfToPlot = dfOutputPlot.reset_index()
    print(dfToPlot)
    dfOutputPlot.to_csv(os.path.join(dirName, '..', 'results', 'shareRewardBaseResult_wolfFlex_sheep6_No5Folder_120000eps_ToPlot', wolfType + '_' + measureLabel +'.csv'))
    
    # import ipdb; ipdb.set_trace()
    #sns.set_style("whitegrid")        # darkgrid(Default), whitegrid, dark, white, ticks
    f, ax = plt.subplots(figsize=(5, 5))
    g = sns.barplot(x='Stag Speed Multiplier', y = measure, hue = 'Set Size', data = dfMeanBySubAndCondtitionResetIndex, estimator=np.mean, ci=68, capsize=.05, errwidth=2, palette = 'Greys')#, color='Red', order = dfSelfAverageScore['targetColorIndexUnshuffled'], palette='Reds')
    # h = sns.histplot(df, x = "Set Size", hue = "Churn", multiple = "stack", stat = "percent")


    ax.set_ylim(0, 80)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    plt.title(wolfType, fontsize=24)
    plt.ylabel(measureLabel, fontsize=24)
    plt.xlabel('Stag Speed Multiplier', fontsize=24)

    # plt.xlabel('singletonOrNot', rotation='horizontal', fontsize=16)
    plt.savefig(os.path.join(dirName, '..', 'results', 'shareRewardBaseResult_wolfFlex_sheep6_No5Folder_120000eps_ToPlot', wolfType + '_' + measureLabel +'.png'), \
                    bbox_inches='tight', dpi = 100, orientation="landscape")
    plt.show()
