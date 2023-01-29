import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
from src.visualization import DrawBackground, DrawNewState, DrawImage, drawText
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld
from pygame.color import THECOLORS
import os
import random
import copy

class NewtonChaseTrialInference():
    def __init__(self, screen, killzone, sheepLife, targetColors, numOfWolves, numOfBlocks, stopwatchEvent,maxTrialStep, wolfActionUpdateInterval, sheepActionUpdateInterval, displayDT,
                 allDrawNewStateFun, recordEaten, getEntityPos, getEntityVel, getEntityCaughtHistory, allWolfPolicy, allSheepPolicy, allTransitFun, 
                 allWolfRewardFun, allGetIntentionDistributions, allRecordActionForUpdateIntention, allResetIntenions, reshapeWolfActionInHybrid):
        self.screen = screen
        self.killzone = killzone
        self.sheepLife = sheepLife
        self.targetColors = targetColors
        self.numOfWolves = numOfWolves
        self.numOfBlocks = numOfBlocks
        self.maxTrialStep = maxTrialStep
        self.wolfActionUpdateInterval = wolfActionUpdateInterval
        self.sheepActionUpdateInterval = sheepActionUpdateInterval
        self.displayDT = displayDT
        self.allDrawNewStateFun = allDrawNewStateFun
        self.stopwatchEvent = stopwatchEvent
        self.recordEaten = recordEaten
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos
        self.getEntityVel = getEntityVel
        self.getEntityCaughtHistory = getEntityCaughtHistory
        self.allWolfPolicy = allWolfPolicy
        self.allSheepPolicy = allSheepPolicy
        self.allTransitFun = allTransitFun
        self.allWolfRewardFun = allWolfRewardFun
        self.allGetIntentionDistributions = allGetIntentionDistributions
        self.allRecordActionForUpdateIntention = allRecordActionForUpdateIntention
        self.allResetIntenions = allResetIntenions
        self.reshapeWolfActionInHybrid = reshapeWolfActionInHybrid
        
    def __call__(self, initState, score, currentStopwatch, trialIndex, trailPickleData):

        trajectoryHuman = trailPickleData['trajectory']
        initState = trajectoryHuman[0][0]
        condition = trailPickleData['condition']
        sheepConcern = condition['sheepConcern']
        blockSize = condition['blockSize']
        sheepMaxSpeed = condition['sheepMaxSpeed']
        
        if blockSize <= 0:
            self.numOfBlocks = 0
        else:
            self.numOfBlocks = 2

        results = co.OrderedDict()
        killZone = self.killzone
        if 'targetColorIndex' in condition.keys():
            condition['targetColorIndexUnshuffled']  = copy.deepcopy(condition['targetColorIndex'])
            targetColorIndexes = condition['targetColorIndex']
            results['targetColorIndexUnshuffled'] = str(targetColorIndexes)
            #random.shuffle(targetColorIndexes)
            sheepNums = len(targetColorIndexes)
            targetColors = [self.targetColors[colorIndex] for colorIndex in targetColorIndexes]
        else:
            targetColorIndexes = 'None'
            condition['targetColorIndex'] = targetColorIndexes
            sheepNums = condition['sheepNums']
            targetColors = random.sample(self.targetColors, sheepNums)

        wolfForce = 5
        sheepForce = wolfForce * condition['sheepWolfForceRatio']

        pickleResults = co.OrderedDict()
        pickleResults['condition'] = condition

        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state, agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                      range(self.numOfWolves, self.numOfWolves + sheepNums)]
        getBlockPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                     range(self.numOfWolves + sheepNums, self.numOfBlocks + self.numOfWolves + sheepNums)]
                                     
        pause = True
        state = initState
        stateList = []
        trajectory = []
        initTargetPositions = getTargetPos(initState)
        initPlayerPositions = getPlayerPos(initState)
        initBlockPositions = getBlockPos(initState)

        results["blockPositions"] = str([[0, 0], [0, 0]])
        if initBlockPositions:
            results["blockPositions"] = str(initBlockPositions)

        # readyTime = 1000
        currentEatenFlag = [0] * len(initTargetPositions)
        currentCaughtHistory = [0] * len(initTargetPositions)
        # while readyTime > 0:
        #     pg.time.delay(32)
        #     self.drawNewState(targetColors, initTargetPositions, initPlayerPositions, initBlockPositions, finishTime, score, currentEatenFlag)
        #     drawText(self.screen, 'ready', THECOLORS['white'],
        #              (self.screen.get_width() * 8 / 3, self.screen.get_height() / 2), 100)
        #     pg.display.update()
        #     readyTime -= self.stopwatchUnit
        initialTime = time.get_ticks()
        eatenFlag = [0] * len(initTargetPositions)
        hunterFlag = [0] * len(initPlayerPositions)
        caughtFlag = [0] * len(initTargetPositions)
        trialStep = -1
        caughtTimes = 0
        rewardList = []
        eatenFlagList = []
        hunterFlagList = []
        caughtFlagList = []
        currentEatenFlagList = []
        currentCaughtFlagList = []
        
        getIntentionDistributions = self.allGetIntentionDistributions[sheepNums, sheepMaxSpeed, blockSize]
        recordActionForUpdateIntention = self.allRecordActionForUpdateIntention[sheepNums, sheepMaxSpeed, blockSize]
        resetIntenions = self.allResetIntenions[sheepNums, sheepMaxSpeed, blockSize]
        resetIntenions()
        
        wolfPolicy = self.allWolfPolicy[sheepNums, sheepMaxSpeed, blockSize]
        sheepPolicy = self.allSheepPolicy[sheepNums, sheepMaxSpeed, blockSize]
        rewardWolf = self.allWolfRewardFun[sheepNums, sheepMaxSpeed]
        transit = self.allTransitFun[sheepNums, sheepMaxSpeed, blockSize]
        drawNewState = self.allDrawNewStateFun[sheepNums, sheepMaxSpeed, blockSize]
        
        replacedIndex = np.random.randint(self.numOfWolves)
        trialStep = -1
        inferenceInterval = 10
        while pause:
            trialStep += 1
            # pg.time.delay(32)
            remainningStep = max(0, self.maxTrialStep - trialStep)
            
            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            # print(trialStep, 'state', state)
            stateHumanTrial, actionHumanTrial, nextStateHumanTrial = trajectoryHuman[trialStep]
            actionWolfHumanTrialReshape = [humanWolfAction for humanWolfAction in actionHumanTrial[0 : self.numOfWolves]] # for shared agency trjaectory
            # actionWolfHumanTrialReshape = [self.reshapeWolfActionInHybrid(humanWolfAction) for humanWolfAction in actionHumanTrial[0 : self.numOfWolves]] 
            actionWolfHumanTrial = [tuple(act) for act in actionWolfHumanTrialReshape]
                
            actionSheepHumanTrial = actionHumanTrial[self.numOfWolves : self.numOfWolves + sheepNums]
                
            action = list(actionWolfHumanTrial) + list(actionSheepHumanTrial)
            # print(trialStep, 'action', action)
            recordActionForUpdateIntention([action])
            if np.mod(trialStep, inferenceInterval) == 0:
                supposedWolfAction = [sampleAction(state) for sampleAction in wolfPolicy] # just for running inferrence and updateIntention in 'sampleAction'
                
            nextState = transit(state, actionWolfHumanTrialReshape, actionSheepHumanTrial, wolfForce, sheepForce)
            reward = rewardWolf(state, action, nextState)[0]
            # print(reward, score)
            score += reward
            rewardList.append(reward)
            
            # pause = self.checkTerminationOfTrial(currentStopwatch)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    currentStopwatch = currentStopwatch + self.stopwatchUnit
            currentEatenFlag, eatenFlag, hunterFlag = self.recordEaten(targetPositions, playerPositions, killZone, eatenFlag, hunterFlag)
            currentCaughtHistory = [self.getEntityCaughtHistory(state, sheepID) for sheepID in range(self.numOfWolves, self.numOfWolves + sheepNums)]
            currentCaughtFlag = (np.array(currentCaughtHistory) == self.sheepLife) + 0
            caughtFlag = np.array(caughtFlag) + np.array(currentCaughtFlag)
            #print(caughtFlag, currentCaughtFlag)
            caughtTimes = np.sum(caughtFlag)
            remainningTime = int(remainningStep * (self.displayDT + 1))
            # drawNewState(targetColors, targetPositions, playerPositions, initBlockPositions, remainningTime, score, currentEatenFlag, currentCaughtHistory)
            pg.display.update()
            eatenFlagList.append(eatenFlag)
            hunterFlagList.append(hunterFlag)
            caughtFlagList.append(caughtFlag)
            currentEatenFlagList.append(currentEatenFlag)
            currentCaughtFlagList.append(currentCaughtFlag)
            
            trajectory.append((state, action, nextState))
            state = nextState
            stateList.append(nextState)
            
            # pause = self.checkTerminationOfTrial(currentStopwatch)
            # if trialStep > self.maxTrialStep or caughtTimes> 0: # or caughtTimes > 0 for terminal game
                # rewardList = [int((self.maxTrialStep - trialStep + 1) * self.displayDT) / 1000]
                # score += sum(rewardList)
            if trialStep > self.maxTrialStep:
                pause = False
            # print(wolfPolicy[0].updateIntention.formerIntentionPriors)
        intentionDistributions = getIntentionDistributions()
        # trajectoryWithIntentionDists = [tuple(list(SASRPair) + list(intentionDist)) for SASRPair, intentionDist in zip(trajectory, intentionDistributions)]
        
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        pickleResults['trajectory'] = trajectory
        # pickleResults['trajectory'] = trajectoryWithIntentionDists
        pickleResults["currentEatenFlagList"] = currentEatenFlagList
        pickleResults["currentCaughtFlagList"] = currentCaughtFlagList
        pickleResults['intentionDistributions'] = intentionDistributions
        results["sheepMaxSpeed"] = sheepMaxSpeed
        results["sheepNums"] = sheepNums
        results["blockSize"] = blockSize
        results["sheepConcern"] = sheepConcern
        results["targetColorIndexes"] = str(targetColorIndexes)
        results["trialTime"] = wholeResponseTime
        results["hunterFlag"] = str(hunterFlag)
        results["sheepEatenFlag"] = str(eatenFlag)
        results["caughtFlag"] = str(list(caughtFlag))
        results["caughtTimes"] = caughtTimes
        results["trialScore"] = sum(rewardList)

        return pickleResults, results, nextState, score, currentStopwatch, eatenFlag

def calculateGridDistance(gridA, gridB):
    return np.linalg.norm(np.array(gridA) - np.array(gridB), ord=2)


def isAnyKilled(humanGrids, targetGrid, killzone):
    return np.any(np.array([calculateGridDistance(humanGrid, targetGrid) for humanGrid in humanGrids]) < killzone)


class RecordEatenNumber:

    def __init__(self, isAnyKilled):
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions, killzone, eatenFlag, hunterFlag):
        currentEatenFlag = [0] * len(targetPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, killzone):
                eatenFlag[i] += 1
                currentEatenFlag[i] = 1
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, killzone):
                hunterFlag[i] += 1
                hunterReward = True
                break
        return currentEatenFlag, eatenFlag, hunterFlag


class CheckEatenVariousKillzone:
    def __init__(self, isAnyKilled):
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions, killzone):
        eatenFlag = [False] * len(targetPositions)
        hunterFlag = [False] * len(playerPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, killzone):
                eatenFlag[i] = True
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, killzone):
                hunterFlag[i] = True
                break
        return eatenFlag, hunterFlag


class CheckTerminationOfTrial:
    def __init__(self, finishTime):
        self.finishTime = finishTime

    def __call__(self, currentStopwatch):
        if currentStopwatch >= self.finishTime:
            pause = False
        else:
            pause = True
        return pause


