# -*- coding: utf-8 -*-
from psychopy import visual, core, event
import os
import sys
import csv
import json
import numpy as np
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from src.writer import loadFromPickle, saveToPickle

def drawCircle(wPtr, pos, size):
    circle = visual.Circle(win=wPtr, lineColor='grey', colorSpace='rgb255', pos=pos, size=size)
    return circle

def showText(wPtr, textHeight, text, position):
    introText = visual.TextStim(wPtr, height=textHeight, font='Times New Roman', text=text, pos=position)
    return introText


def expandCoordination(position, expandRatio):
    a = len(position[0])
    newPos = []
    for i in position:
        newPos.append([i[j] * expandRatio for j in range(a)])
    return newPos


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
    for i in reader:
        a.append(json.loads(i[keyName]))
    f.close()
    return a


def main():
    textHeight = 24
    expandRatio = 300 * 0.95
    wolfRadius = 0.065
    sheepRadius = 0.065
    objectSize = wolfRadius * 2 * expandRatio
    targetSize = sheepRadius * 2 * expandRatio
    #waitTime = 0.085 # for model demo
    waitTime = 0.03  # for human demo

    wPtr = visual.Window(size=[610, 610], units='pix', fullscr=False)
    myMouse = event.Mouse(visible=True, win=wPtr)
    introText = showText(wPtr, textHeight, 'Press ENTER to start', (0, 250))
    introText.autoDraw = True
    wPtr.flip()
    keys = []
    while 'return' not in keys:
        keys = event.getKeys()
    introText.autoDraw = False
    timerText = showText(wPtr, 50, u'', (0, 0))  # 每个trial间的注视点

    dirName = os.path.dirname(__file__)
    numWolves = 3
    # numSheep = 4
    # numAIWolves = 1
    # actionAIIntervel = 1
    # cov = 0.5
    # priorDecayRate = 0.7
    # pickleName = 'ModelType=SA_SaveName=cov' + str(cov) + '_priorDecayRate=' + str(priorDecayRate) + '_sheepNums=' + str(numSheep) + '.pickle'
    # fileName = os.path.join(dirName, '..', 'results', 'machineResultsWithIntention', 'SA', pickleName)
    pickleName = 'Rs9.pickle'
    fileName = os.path.join(dirName, '..', 'results', pickleName)
    allRawData = loadFromPickle(fileName)

    endTrial = len(allRawData)
    startTrial = 0
    stepForLoop = 1
    allTrialList = list(range(startTrial, endTrial, stepForLoop))
    print(allTrialList[-1])
    targetChoiceList = []
    stepCountList = []
    targetNumList = []
    blockSizeList = []
    print(pickleName, len(allRawData))

    for i in allTrialList:
        wPtr.flip()
        print('trial', i)
        #targetTrajKeyName = 'sheeps traj'
        #targetNumListKeyName = 'sheepNums'
        #targetPos, targetNumList = readFun(targetTrajKeyName), readFun(targetNumListKeyName)
        oneTrialRawData = allRawData[i]
        trajectory = oneTrialRawData['trajectory']
        condition = oneTrialRawData['condition']
        numSheep = condition['sheepNums']
        targetNumList.append(numSheep)

        blockSize = condition['blockSize']
        obstacleSizeExpanded = blockSize * 2 * expandRatio
        blockSizeList.append(blockSize)

        if blockSize <= 0:
            numBlocks = 0
        else:
            numBlocks = 2

        if (blockSize != 0) or (numSheep != 1):
        # if i not in [11, 18]:
            continue

        #import ipdb; ipdb.set_trace()
        # -----position pre-processing-----
        targetPoses = [[], [], [], []]
        playerPoses = [[], [], []]
        blockPoses = [[], []]
        targetProbabilities = [[], [], [], []]
        print(len(trajectory))
        for timeStep in trajectory[:-20]:
            # state, action, nextState, inferredProbability = timeStep
            state, action, nextState = timeStep
            for wolfIndex in range(0, numWolves):
                playerPoses[wolfIndex].append(state[wolfIndex][0:2])
            for sheepIndex in range(numWolves, numWolves + numSheep):
                targetPoses[sheepIndex - numWolves].append(state[sheepIndex][0:2])
                # sumOfTargetProbabilities = [list(inferredProbability[wolfIndex].values()) for wolfIndex in range(numWolves)]
                # targetProbabilities[sheepIndex - numWolves].append(np.mean(sumOfTargetProbabilities, axis = 0)[sheepIndex - numWolves])
            for blockIndex in range(numWolves + numSheep, numWolves + numSheep + numBlocks):
                blockPoses[blockIndex - numWolves - numSheep].append(state[blockIndex][0:2])

        targetPos1, targetPos2, targetPos3, targetPos4 = targetPoses
        playerPos1, playerPos2, playerPos3 = playerPoses
        blockPos1, blockPos2 = blockPoses

        # targetProb1, targetProb2, targetProb3, targetProb4 = targetProbabilities
        # print(playerPoses[0])
        # print(targetPoses[0])
        # import ipdb;ipdb.set_trace()
        # drawFunction Set


    wPtr.flip()
    event.waitKeys()
    wPtr.close()


if __name__ == "__main__":
    main()
