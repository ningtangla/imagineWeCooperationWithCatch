import pickle
import numpy as np
import pandas as pd


class NewtonExperimentWithResetIntentionHybridTeam():
    def __init__(self, restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset, drawImage, writable = False):
        self.trial = trial
        self.writer = writer
        self.pickleWriter = pickleWriter
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.hasRest = hasRest
        self.writable = writable

    def __call__(self, trailsRawData, restTimes):
        trialIndex = 0
        score = 0
        pickleDataList = []
        for humanTrail in trailsRawData:
            condition = humanTrail['condition']
            if 'targetColorIndex' in condition.keys():
                targetColorIndexes = condition['targetColorIndex']
                sheepNums = len(targetColorIndexes)
            else:
                sheepNums = condition['sheepNums']
            blockSize = condition['blockSize']
            sheepConcern = condition['sheepConcern']
            initState = self.reset(sheepNums, blockSize)
            currentStopwatch = 0
            pickleResult, result, finalState, score, currentStopwatch, eatenFlag = self.trial\
                (initState, score, currentStopwatch, trialIndex, humanTrail)
            result["totalScore"] = str(score)
            pickleResult['trialIndex'] = trialIndex
            response = self.experimentValues.copy()
            response.update(result)
            pickleDataList.append(pickleResult)
            trialIndex += 1
            if self.writable:
                self.writer(response, trialIndex)
            totalTrialNum = len(trailsRawData)
            if np.mod(trialIndex, totalTrialNum/restTimes) == 0 and self.hasRest and (trialIndex < totalTrialNum):
                self.drawImage(self.restImage)
        if self.writable:
            self.pickleWriter(pickleDataList)

class NewtonExperimentWithDiffBlocksHybridTeam():
    def __init__(self, restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset, drawImage, writable = False):
        self.trial = trial
        self.writer = writer
        self.pickleWriter = pickleWriter
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.hasRest = hasRest
        self.writable = writable

    def __call__(self, finishTime, trailsRawData, restTimes):
        trialIndex = 0
        score = 0
        # for trialIndex in range(len(trialCondtions)):
        pickleDataList = []
        for humanTrail in trailsRawData:
            condition = humanTrail['condition']
            if 'targetColorIndex' in condition.keys():
                targetColorIndexes = condition['targetColorIndex']
                sheepNums = len(targetColorIndexes)
            else:
                sheepNums = condition['sheepNums']
            blockSize = condition['blockSize']
            sheepConcern = condition['sheepConcern']
            initState = self.reset(sheepNums, blockSize)
            currentStopwatch = 0
            pickleResult, result, finalState, score, currentStopwatch, eatenFlag = self.trial\
                (initState, score, finishTime, currentStopwatch, trialIndex, humanTrail)
            result["totalScore"] = str(score)
            pickleResult['trialIndex'] = trialIndex
            response = self.experimentValues.copy()
            response.update(result)
            pickleDataList.append(pickleResult)
            trialIndex += 1
            if self.writable:
                self.writer(response, trialIndex)
            totalTrialNum = len(trailsRawData)
            if np.mod(trialIndex, totalTrialNum/restTimes) == 0 and self.hasRest and (trialIndex < totalTrialNum):
                self.drawImage(self.restImage)
        if self.writable:
            self.pickleWriter(pickleDataList)

class NewtonExperiment():
    def __init__(self,restImage,hasRest, trial, writer,pickleWriter, experimentValues, reset, drawImage):
        self.trial = trial
        self.writer = writer
        self.pickleWriter = pickleWriter
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.hasRest = hasRest

    def __call__(self, finishTime, trialCondtions, restTimes):
        trialIndex = 0
        score = 0
        # for trialIndex in range(len(trialCondtions)):
        pickleDataList = []
        for condition in trialCondtions:
            # condition = trialCondtions[trialIndex]
            print('trial', trialIndex + 1)
            print(condition)
            sheepNums = condition['sheepNums']
            initState = self.reset(sheepNums)
            currentStopwatch = 0
            pickleResult, result, finalState, score, currentStopwatch, eatenFlag = self.trial\
                (initState, score, finishTime, currentStopwatch, trialIndex, condition)
            result["sheepNums"] = sheepNums
            result["totalScore"] = str(score)
            pickleResult['trialIndex'] = trialIndex
            response = self.experimentValues.copy()
            response.update(result)
            pickleDataList.append(pickleResult)
            trialIndex += 1
            self.writer(response, trialIndex)
            totalTrialNum = len(trialCondtions)
            if np.mod(trialIndex, totalTrialNum/restTimes) == 0 and self.hasRest and (trialIndex < totalTrialNum):
                self.drawImage(self.restImage)
        self.pickleWriter(pickleDataList)


class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, finishTime, trialCondtions):

        trialIndex = 0
        score = np.array([0, 0])

        trialNum = 4
        blockResult=[]
        for conditon in trialCondtions:
            sheepNums = conditon['sheepNums']
            targetPositions, playerGrid = self.initialWorld(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trialIndex', trialIndex)
            # response = self.experimentValues.copy()
            traj, targetPositions, playerGrid, score, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                targetPositions, playerGrid, score, currentStopwatch, trialIndex, timeStepforDraw, sheepNums)
            # response.update(results)

            blockResult.append({'sheepNums': sheepNums, 'score': score, 'traj': traj })

            if currentStopwatch >= finishTime:
                break
            targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
        self.writer(blockResult,self.resultsPath)
        return blockResult


        
class ExperimentServer():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.resultsPath = resultsPath

    def __call__(self, finishTime, trialCondtions):

        trialIndex = 0
        score = np.array([0, 0])

        trialNum = 4
        blockResult=[]
        for conditon in trialCondtions:
            sheepNums = conditon['sheepNums']
            targetPositions, playerGrid = self.initialWorld(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trialIndex', trialIndex)

            traj, targetPositions, playerGrid, score, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                targetPositions, playerGrid, score, currentStopwatch, trialIndex, timeStepforDraw, sheepNums)

            blockResult.append({'sheepNums': sheepNums, 'score': score, 'traj': traj })

            if currentStopwatch >= finishTime:
                break
            targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
        self.writer(blockResult,self.resultsPath)
        print(blockResult)
        return blockResult