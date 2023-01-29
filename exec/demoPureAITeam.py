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
    objectSize = 30
    targetSize = 20
    waitTime = 0.085 # for model demo
    #waitTime = 0.03  # for human demo

    wPtr = visual.Window(size=[768, 768], units='pix', fullscr=False)
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
    numSheep = 4
    numAIWolves = 1
    actionAIIntervel = 1
    cov = 0.5
    priorDecayRate = 0.7
    pickleName = 'ModelType=SA_SaveName=cov' + str(cov) + '_priorDecayRate=' + str(priorDecayRate) + '_sheepNums=' + str(numSheep) + '.pickle'
    fileName = os.path.join(dirName, '..', 'results', 'machineResultsWithIntention', 'SA', pickleName)
    trajectories =loadFromPickle(fileName)

    trialNum = 20
    startTrial = 0
    targetChoiceList = []
    stepCountList = []
    targetNum = []
    print(pickleName, len(trajectories))
    for i in range(startTrial, trialNum):
        print('trial', i + 1)
        #targetTrajKeyName = 'sheeps traj'
        #targetNumKeyName = 'sheepNums'
        #targetPos, targetNum = readFun(targetTrajKeyName), readFun(targetNumKeyName)
        trajectory = trajectories[i]['trajectory']
        targetNum.append(numSheep)

        #import ipdb; ipdb.set_trace()
        # -----position pre-processing-----
        targetPoses = [[], [], [], []]
        playerPoses = [[], [], []]
        targetProbabilities = [[], [], [], []]
        for timeStep in trajectory:
            state, action, nextState, inferredProbability = timeStep
            for wolfIndex in range(0, numWolves):
                playerPoses[wolfIndex].append(state[wolfIndex][0:2])
            for sheepIndex in range(numWolves, numWolves + numSheep):
                targetPoses[sheepIndex - numWolves].append(state[sheepIndex][0:2])
                sumOfTargetProbabilities = [list(inferredProbability[wolfIndex].values()) for wolfIndex in range(numWolves)]
                targetProbabilities[sheepIndex - numWolves].append(np.mean(sumOfTargetProbabilities, axis = 0)[sheepIndex - numWolves])
                #print(wolfIndex, state[wolfIndex][0:2])

        targetPos1, targetPos2, targetPos3, targetPos4 = targetPoses
        playerPos1, playerPos2, playerPos3 = playerPoses
        targetProb1, targetProb2, targetProb3, targetProb4 = targetProbabilities
        #print(playerPoses[0])
        #print(targetPoses[0])
        #import ipdb;ipdb.set_trace()
        # drawFunction Set
        expandRatio = 300

        targetPos1 = expandCoordination(targetPos1, expandRatio)
        drawTargetCircleFun = lambda pos: drawCircle(wPtr, pos[0], targetSize)
        target1Traj = drawTargetCircleFun(targetPos1)
        if targetNum[i] == 2:
            targetPos2 = expandCoordination(targetPos2, expandRatio)
            target2Traj = drawTargetCircleFun(targetPos2)
        #import ipdb; ipdb.set_trace()

        if targetNum[i] == 4:
            targetPos2 = expandCoordination(targetPos2, expandRatio)
            targetPos3 = expandCoordination(targetPos3, expandRatio)
            targetPos4 = expandCoordination(targetPos4, expandRatio)
            target1Traj, target2Traj, target3Traj, target4Traj = drawTargetCircleFun(targetPos1), drawTargetCircleFun(targetPos2), drawTargetCircleFun(targetPos3), drawTargetCircleFun(targetPos4)

        expandFun = lambda pos: expandCoordination(pos, expandRatio)
        playerPos1, playerPos2, playerPos3, = expandFun(playerPos1), expandFun(
            playerPos2), expandFun(playerPos3)
        drawPlayerCircleFun = lambda pos: drawCircle(wPtr, pos[0], objectSize)
        player1Traj, player2Traj, player3Traj = drawPlayerCircleFun(playerPos1), drawPlayerCircleFun(
            playerPos2), drawPlayerCircleFun(playerPos3)

        # only color the players
        # setColorFun = lambda traj: traj.setFillColor('white')
        # setColorFun(player1Traj)
        # setColorFun(player2Traj)
        # setColorFun(player3Traj)
        #setColorFun(target1Traj)
        #if targetNum[i] == 2:
        #    setColorFun(target2Traj)
        #if targetNum[i] == 4:
        #    setColorFun(target2Traj)
        #    setColorFun(target3Traj)
        #    setColorFun(target4Traj)

        player1Traj.setFillColor('red')
        player2Traj.setFillColor('blue')
        player3Traj.setFillColor('green')

        # color all the objects for demo
        # target1Traj.setFillColor('orange')
        # if targetNum[i] == 2:
        #     target2Traj.setFillColor('DarkOrange')
        # if targetNum[i] == 4:
        #     target2Traj.setFillColor('DarkOrange')
        #     target3Traj.setFillColor('SandyBrown')
        #     target4Traj.setFillColor('goldenrod')
        # 'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)

        player1Traj.autoDraw = True
        player2Traj.autoDraw = True
        player3Traj.autoDraw = True
        target1Traj.autoDraw = True
        if targetNum[i] == 2:
            target2Traj.autoDraw = True
        if targetNum[i] == 4:
            target2Traj.autoDraw = True
            target3Traj.autoDraw = True
            target4Traj.autoDraw = True

        stepCount = 0
        if targetNum[i] == 1:
            for x, y, z, a, b in zip(playerPos1, playerPos2, playerPos3, targetPos1, targetProb1):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (b - 0))
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    # print('press:', keys)
                    break

        if targetNum[i] == 2:
            for x, y, z, a, b, c, d in zip(playerPos1, playerPos2, playerPos3, targetPos1, targetPos2, targetProb1, targetProb2):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                target2Traj.setPos(b)
                probs = np.array([c, d])
                probBiggestIndex = np.argmax(probs)
                if probBiggestIndex == 0:
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.5 + 0.01 * min(50, stepCount)))
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.5 - 0.01 * min(50, stepCount)))
                if probBiggestIndex == 1:
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.5 + 0.01 * min(50, stepCount)))
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.5 - 0.01 * min(50, stepCount)))
                #print(c, d)
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    while True:
                        if myMouse.isPressedIn(target1Traj):
                            choice = 'target1'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target2Traj):
                            choice = 'target2'
                            targetChoiceList.append(choice)
                            break
                    break

        if targetNum[i] == 4:
            for x, y, z, a, b, c, d, e, f, g, h in zip(playerPos1, playerPos2, playerPos3, targetPos1, targetPos2, targetPos3, targetPos4, targetProb1, targetProb2, targetProb3, targetProb4):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                target2Traj.setPos(b)
                target3Traj.setPos(c)
                target4Traj.setPos(d)
                probs = np.array([e, f, g, h])
                probBiggestIndex = np.argmax(probs)
                if probBiggestIndex == 0:
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.25 + 0.01 * min(75, stepCount)))
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target3Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target4Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                if probBiggestIndex == 1:
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.25 + 0.01 * min(75, stepCount)))
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target3Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target4Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                if probBiggestIndex == 2:
                    target3Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.25 + 0.01 * min(75, stepCount)))
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target4Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                if probBiggestIndex == 3:
                    target4Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.25 + 0.01 * min(75, stepCount)))
                    target2Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target3Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)
                    target1Traj.setFillColor(np.array([55, 55, 55]) + np.array([200, 200, 0]) * (0.75 - 0.01 * min(75, stepCount))/3)

                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    while True:
                        if myMouse.isPressedIn(target1Traj):
                            choice = 'target1'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target2Traj):
                            choice = 'target2'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target3Traj):
                            choice = 'target3'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target4Traj):
                            choice = 'target4'
                            targetChoiceList.append(choice)
                            break
                    break

        # print('choice:', choice)
        print('stop step:', stepCount)
        stepCountList.append(stepCount)
        player1Traj.autoDraw = False
        player2Traj.autoDraw = False
        player3Traj.autoDraw = False
        target1Traj.autoDraw = False
        if targetNum[i] == 2:
            target2Traj.autoDraw = False
        if targetNum[i] == 4:
            target2Traj.autoDraw = False
            target3Traj.autoDraw = False
            target4Traj.autoDraw = False

        restTime = 5
        restDuration = (trialNum-startTrial)/restTime
        if np.mod((i-startTrial+1), restDuration) != 0:
            dtimer = core.CountdownTimer(1)  # wait for 1s
            while dtimer.getTime() > 0:
                timerText.text = '+'
                timerText.bold = True
                timerText.draw()
            wPtr.flip()
        else:   # rest
            restText = showText(wPtr, textHeight, 'Press Space to continue', (0, 300))
            restText.autoDraw = True
            wPtr.flip()
            event.waitKeys()
            restText.autoDraw = False

    wPtr.flip()
    event.waitKeys()
    wPtr.close()


if __name__ == "__main__":
    main()
