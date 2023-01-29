# -*- coding: utf-8 -*-
from psychopy import visual, core, event
import os
import sys
import csv
import json
import numpy as np
import copy
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from src.writer import loadFromPickle, saveToPickle

def drawCircle(wPtr, pos, size):
    circle = visual.Circle(win=wPtr, lineColor='grey', colorSpace='rgb255', pos=pos, size=size)
    return circle

def showText(wPtr, textHeight, text, position):
    introText = visual.TextStim(wPtr, height=textHeight, font='Times sheep Roman', text=text, pos=position)
    return introText


def expandCoordination(position, expandRatio):
    a = len(position[0])
    sheepPos = []
    for i in position:
        sheepPos.append([i[j] * expandRatio for j in range(a)])
    return sheepPos


def readCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for i in reader:
        a.append(i[keyName])
    f.close()
    return a


def readListCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for z in reader:
        a.append(json.loads(z[keyName]))
    f.close()
    return a


def main():
    textHeight = 24
    expandRatio = 300 * 0.95
    wolfRadius = 0.065
    sheepRadius = 0.065
    objectSize = wolfRadius * 2 * expandRatio
    targetSize = sheepRadius * 2 * expandRatio
    targetColorsOrig = ['orange', 'purple']
    #waitTime = 0.085 # for model demo
    waitTime = 0.05  # for human demo

    dirName = os.path.dirname(__file__)
    fileFolder = os.path.join(dirName, '..', 'results', 'testResult')
    prefixList = []
    allFiles = os.listdir(fileFolder)
    for j in allFiles:
        if os.path.splitext(j)[1] == '.pickle':
            prefixList.append(os.path.splitext(j)[0])
    
    print(prefixList)
    
    eatenLengthAsOneCatch = 20
    
    dfList = []
    for prefix in prefixList:
        numWolves = 3
        fileName = os.path.join(dirName, '..', 'results', 'testResult', prefix + '.pickle')
        allRawData = loadFromPickle(fileName)

        sheepPickleData = []
        endTrial = len(allRawData)
        startTrial = 0
        stepForLoop = 1
        allTrialList = list(range(startTrial, endTrial, stepForLoop))
        print(allTrialList[-1])
        targetChoiceList = []
        stepCountList = []
        targetNumList = []
        blockSizeList = []
        sheepMaxSpeedList = []
        oneSubjectAtLeastOneCatch = []

        # print(prefix, len(allRawData))
        csvFileName = os.path.join(dirName, '..', 'results', 'testResult', prefix + '.csv')
        colorIndexFromCSV = readListCSV(csvFileName, 'targetColorIndexes')
        colorIndexUnshuffledFromCSV = readListCSV(csvFileName, 'targetColorIndexUnshuffled')
        allRawDatasheep = copy.deepcopy(allRawData)
        
        df = pd.DataFrame()
        for i in allTrialList:
            print('trial', i)
            #targetTrajKeyName = 'sheeps traj'
            #targetNumListKeyName = 'sheepNums'
            #targetPos, targetNumList = readFun(targetTrajKeyName), readFun(targetNumListKeyName)
            oneTrialRawData = copy.deepcopy(allRawData[i])
            trajectory = oneTrialRawData['trajectory']
            currentEatenFlagList = oneTrialRawData['currentEatenFlagList']
            condition = oneTrialRawData['condition']
            numSheep = condition['sheepNums']
            sheepMaxSpeed = condition['sheepMaxSpeed']
            blockSize = condition['blockSize']
            obstacleSizeExpanded = blockSize * 2 * expandRatio
            blockSizeList.append(blockSize)

            if blockSize <= 0:
                numBlocks = 0
            else:
                numBlocks = 2
                
            if ('targetColorIndex' in condition.keys()) & (condition['targetColorIndex'] != 'None'):

                # condition.update({'targetColorIndex': list(colorIndexFromCSV[i])})
                # condition.update({'targetColorIndexUnshuffled': list(colorIndexUnshuffledFromCSV[i])})
                targetColorIndex = condition['targetColorIndex'] 
                targetColor = [targetColorsOrig[index] for index in targetColorIndex]
                numSheep = len(targetColorIndex)
                targetNumList.append(numSheep)
                # if (targetColorIndex != [0, 0, 1]) and (targetColorIndex != [0, 1, 1]):
                    # continue
            sheepMaxSpeedList.append(sheepMaxSpeed)
            
            print('len', len(currentEatenFlagList))
            teamTargetNumCountTimestep = np.zeros(numWolves)
            atLeastOneCatch = 0
            for timeIndex in range(len(currentEatenFlagList)):
                # state, action, nextState, inferredProbability = timeStep
                if timeIndex > eatenLengthAsOneCatch:
                    # print(np.array(currentEatenFlagList)[timeIndex - eatenLengthAsOneCatch: timeIndex, :])
                    catchOrNot = int(np.sum(np.array(currentEatenFlagList)[timeIndex - eatenLengthAsOneCatch: timeIndex, : ]) == eatenLengthAsOneCatch)
                    if catchOrNot == 1:
                        atLeastOneCatch = 1
                        break
            oneSubjectAtLeastOneCatch.append(atLeastOneCatch)
            # print(np.array(oneSubjectTeamDistribution), targetNumList)
        df['oneSubjectAtLeastOneCatch'] = np.array(oneSubjectAtLeastOneCatch)

        df['sheepNums'] = np.array(targetNumList)
        df['sheepMaxSpeed'] =  np.array(sheepMaxSpeedList)
        df['singletonOrNot'] = [ 1 - (np.sum(colorIndex) == len(colorIndex) or np.sum(colorIndex) == 0)   for colorIndex in colorIndexUnshuffledFromCSV]
        dfList.append(df)
        
    finalDf = pd.concat(dfList)
    dfToPlot = finalDf[finalDf['sheepNums'] == 1]
    dfMelted = pd.melt(dfToPlot, id_vars= ['sheepMaxSpeed', 'singletonOrNot', 'sheepNums'], var_name='teamNum', value_name = 'oneSubjectAtLeastOneCatch')
    # import ipdb;ipdb.set_trace()
    g = sns.barplot(x = 'teamNum', y = 'oneSubjectAtLeastOneCatch', hue = 'singletonOrNot', data = dfMelted, estimator=np.mean, ci=95, capsize=.05, errwidth=2,palette='Reds')# color='Red')#, hue = 'singletonOrNot', order = dfSelfAverageScore['targetColorIndexesUnshuffled'])#, palette='Reds')
    

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.ylabel('percentageOfTimeSteps', fontsize=16)

    # plt.xlabel('singletonOrNot', rotation='horizontal', fontsize=16)

    plt.show()


if __name__ == "__main__":
    main()
