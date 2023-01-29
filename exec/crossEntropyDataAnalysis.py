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
    csvHumanList = []
    csvHybridTeamList = []
    nameList = []
    a = os.listdir(fileFolder)
    for j in a:
        if os.path.splitext(j)[1] == '.csv':
            csvList.append(fileFolder+'/'+j)
            nameList.append(fileFolder
            
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
        

    dfTrialData = pd.concat(allDf)
    dfTrialData= splitListColom(dfTrialData, ['sheepEatenFlag', 'caughtFlag'])#, 'sheepEatenFlagSortBySingleton', 'sheepCatchFlagSortBySingleton', 'sheepEatenFlagPercentageSortBySingleton', 'sheepCatchFlagPercentageSortBySingleton'])
    # dfTrialData= splitListColom(dfTrialData, ['sheepEatenFlag', 'caughtFlag', 'sheepEatenFlagSortBySingleton', 'sheepCatchFlagSortBySingleton', 'sheepEatenFlagPercentageSortBySingleton', 'sheepCatchFlagPercentageSortBySingleton'])
    dfData = copy.deepcopy(dfTrialData[(dfTrialData['sheepNum'] <= 4) & (dfTrialData['caughtTimes'] <= 135)])# & (dfTrialData['sheepNum'] == 3) & (dfTrialData['singletonOrNot'] == 1)])
    dfData['sheep Max Speed'] = dfData['sheepMaxSpeed']
    print(len(dfData.index))
    print(dfData['targetColorIndexUnshuffled'])
    #dfData = dfTrialData[dfTrialData['caughtTimes'] <= 25]
    groupedData = dfData.groupby(['name', 'sheepNum', 'sheep Max Speed'])#, 'blockSize', 'targetColorIndexUnshuffled', 'singletonOrNot'])
    # print(groupedData, dfData['targetColorIndexUnshuffled'])
    dfOutputStat = groupedData['trialScore'].agg(['mean', 'std'])
    dfOutputStat['SE'] = dfOutputStat['std']/np.sqrt(len(allDf))
    dfToStat = dfOutputStat.reset_index()
    print(dfToStat)
    dfOutputStat.to_csv(os.path.join(dirName, '..', 'results', 'trialHuman.csv'))
    
    #dfTotalScore = groupedData.sum()  # total score for every condition
    dfMeanBySubAndCondtition = groupedData.mean()  # average score for every condition
    dfGroupByCondtitionBaseSub = dfMeanBySubAndCondtition.groupby(['sheepNum', 'sheep Max Speed'])
    dfOutputPlot = dfGroupByCondtitionBaseSub['trialScore'].agg(['mean', 'std'])
    dfOutputPlot['SE'] = dfOutputPlot['std']/np.sqrt(len(allDf))
    dfToPlot = dfOutputPlot.reset_index()
    print(dfToPlot)
    dfOutputPlot.to_csv(os.path.join(dirName, '..', 'results', 'trialScoreHuman.csv'))
    # import ipdb; ipdb.set_trace()
    #sns.set_style("whitegrid")        # darkgrid(Default), whitegrid, dark, white, ticks
    f, ax = plt.subplots(figsize=(5, 5))
    # barplot: Default: np.mean
    g = sns.barplot(x='sheepNum', y='trialScore', hue = 'sheep Max Speed', data=dfMeanBySubAndCondtition.reset_index(), estimator=np.mean, ci=68, capsize=.05, errwidth=2, palette = 'Greys')#, color='Red', order = dfSelfAverageScore['targetColorIndexUnshuffled'], palette='Reds')
    # g = sns.barplot( y='isSingletonCatchMost', data=dfData, estimator=np.mean, ci=95, capsize=.05, errwidth=2, color='Red')
    # sns.barplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData, estimator=np.mean, ci=95, capsize=.05, errwidth=2, palette='Blues')
    # sns.boxplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData)
    # dfToPlot = dfData[['sheepEatenFlagPercentageSortBySingleton0',  'sheepEatenFlagPercentageSortBySingleton1', 'sheepEatenFlagPercentageSortBySingleton2', 'sheepEatenFlagPercentageSortBySingleton3','sheepEatenFlagPercentageSortBySingleton4', 'singletonOrNot']]
    # dfToPlot = dfData[['sheepCatchFlagPercentageSortBySingleton0',  'sheepCatchFlagPercentageSortBySingleton1', 'sheepCatchFlagPercentageSortBySingleton2', 'sheepCatchFlagPercentageSortBySingleton3', 'sheepCatchFlagPercentageSortBySingleton4','singletonOrNot']]
    # dfToPlot.columns = ['singleton', 'notSingleton1', 'notSingleton2', 'notSingleton3', 'notSingleton4', 'singletonOrNot']
    # dfMelted = pd.melt(dfToPlot, id_vars= 'singletonOrNot', var_name='targetKind', value_name = 'percentage')
    # print(dfToPlot.groupby(['singletonOrNot']).mean())
    # g = sns.barplot(x = 'targetKind', y = 'percentage', data = dfMelted, estimator=np.mean, ci=95, capsize=.05, errwidth=2, color='Red')#, hue = 'singletonOrNot', order = dfSelfAverageScore['targetColorIndexesUnshuffled'])#, palette='Reds')
    # g.set(xlabel = None)
    
    #for index, row in dfSelfAverageScore.iterrows():
    #    g.text(index-3, row['trialScore']+2, round(row['trialScore'], 2), color="black", ha="center")

    ax.set_ylim(0, 80)

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=20)


    plt.ylabel('Accumulated Reward', fontsize=32)
    plt.xlabel('Num Sheep', fontsize=32)

    # plt.xlabel('singletonOrNot', rotation='horizontal', fontsize=16)

    plt.show()
