import sys
import os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

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
import scipy.stats as stats
from src.writer import loadFromPickle, saveToPickle
from src.mathTools.distribution import sampleFromDistribution,  SoftDistribution, BuildGaussianFixCov, sampleFromContinuousSpace

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
     
class MeasureEntropy:
    def __init__(self, imaginedWeIds, priorIndex):
        self.imaginedWeIds = imaginedWeIds
        self.priorIndex = priorIndex

    def __call__(self, oneTrialPickleData):
        priors = oneTrialPickleData['intentionDistributions']
        # import ipdb; ipdb.set_trace()
        meanEntropyWholeTrajectory = [np.mean([stats.entropy(list(prior[0][agentId].values())) for agentId in self.imaginedWeIds])
                for prior in priors] 
        return meanEntropyWholeTrajectory


class MeasureCrossEntropy:
    def __init__(self, imaginedWeIds, priorIndex):
        self.baseId, self.nonBaseIds = imaginedWeIds
        self.priorIndex = priorIndex

    def __call__(self, oneTrialPickleData):
        priors = oneTrialPickleData['intentionDistributions']
        baseDistributions = [list(prior[0][self.baseId].values())
                for prior in priors] 
        nonBaseDistributionsAllNonBaseAgents = [[list(prior[0][nonBaseId].values()) 
            for nonBaseId in self.nonBaseIds] for prior in priors]
        # trajectory = pickleOriginal[trialIndex]['trajectory']
        # priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        # baseDistributions = [list(prior[self.baseId].values())
                # for prior in priors] 
        # nonBaseDistributionsAllNonBaseAgents = [[list(prior[nonBaseId].values()) 
            # for nonBaseId in self.nonBaseIds] for prior in priors]
        crossEntropies = [stats.entropy(baseDistribution) + np.mean([stats.entropy(baseDistribution, nonBaseDistribution) 
            for nonBaseDistribution in nonBaseDistributions]) 
            for baseDistribution, nonBaseDistributions in zip(baseDistributions, nonBaseDistributionsAllNonBaseAgents)]
        return crossEntropies

class MeasureConvergeRate:
    def __init__(self, imaginedWeIds, priorIndex, chooseIntention):
        self.imaginedWeIds = imaginedWeIds
        self.priorIndex = priorIndex
        self.chooseIntention = chooseIntention

    def __call__(self, oneTrialPickleData):
        # trajectory = pickleOriginal[trialIndex]['trajectory']
        # priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        # intentionSamplesTrajectory = [[[self.chooseIntention(prior[agentId])[0] for agentId in self.imaginedWeIds] 
            # for _ in range(50)] 
            # for prior in priors]
        priors = oneTrialPickleData['intentionDistributions']
        intentionSamplesTrajectory = [[[self.chooseIntention(prior[0][agentId])[0] for agentId in self.imaginedWeIds] 
            for _ in range(50)] 
            for prior in priors]
        convergeRates = [np.mean([np.all([intentionEachAgent == intentionSample[0] for intentionEachAgent in intentionSample])
            for intentionSample in intentionSamples]) 
            for intentionSamples in intentionSamplesTrajectory]
        return convergeRates
        
if __name__=="__main__":


    
    dirName = os.path.dirname(__file__)
    allDf = []
    wolfTypeList = ['Human', 'RS Model', 'IW Model']
    for wolfType in wolfTypeList:
        fileFolder = os.path.join(dirName, '..', 'results', 'InferIWIntention' + wolfType)
        fileNameList = []
        a = os.listdir(fileFolder)
        for j in a:
            if os.path.splitext(j)[1] == '.csv':
                fileName = os.path.splitext(j)[0]
                fileNameList.append(fileName)

        # targetColorIndexesKey = 'targetColorIndexes'
        # targetColorIndexesUnshuffledKey = 'targetColorIndexUnshuffled'
        # sheepEatenFlagKey = 'sheepEatenFlag'
        # sheepCatchFlagKey = 'caughtFlag'
        
        for name in fileNameList:
            csvName = os.path.join(fileFolder, name + '.csv')
            pickleName = os.path.join(fileFolder, name + '.pickle')
            print(pickleName)
            dfFromCSV = pd.read_csv(csvName)
            numTrial = dfFromCSV.shape[0] # numTrial = numRows
            numInferStep = 39
            dfOriginal = pd.DataFrame(np.repeat(dfFromCSV.values, numInferStep, axis=0))
            dfOriginal.columns = dfFromCSV.columns
            dfOriginal['Time Step'] = np.array([list(range(numInferStep))] * numTrial).flatten()

            # readFun = lambda key: readListCSV(csvName, key)
            # targetColorIndexes = readFun(targetColorIndexesKey)
            # targetColorIndexUnshuffled = readFun(targetColorIndexesUnshuffledKey)
            # sheepEatenFlagRawData = readFun(sheepEatenFlagKey)
            # sheepCatchFlagRawData = readFun(sheepCatchFlagKey)
            
            entropyAllTrails = []
            crossEntropyAllTrails = []
            convergeRateAllTrails = []
            pickleOriginal = loadFromPickle(pickleName)

            for trialIndex in range(len(pickleOriginal)):
                oneTrialPickleData = pickleOriginal[trialIndex]
                condition = pickleOriginal[trialIndex]['condition']
                numOfSheep = condition['sheepNums']
                numOfWolves = 3
                baseId = 0
                imaginedWeIds = list(range(numOfWolves))
                imaginedWeIdsForCrossEntropy = [0, list(range(1, numOfWolves))]
                priorIndex = 3
                measureEntropy = MeasureEntropy(imaginedWeIds, priorIndex)
                measureCrossEntropy = MeasureCrossEntropy(imaginedWeIdsForCrossEntropy, priorIndex)
                chooseIntention = sampleFromDistribution
                measureConvergeRate = MeasureConvergeRate(imaginedWeIds, priorIndex, chooseIntention)
                # entropy = measureEntropy(oneTrialPickleData)
                # entropyAllTrails.append(entropy)
                crossEntropy = measureCrossEntropy(oneTrialPickleData)
                crossEntropyAllTrails.append(crossEntropy)
                # convergeRate = measureConvergeRate(oneTrialPickleData)
                # convergeRateAllTrails.append(convergeRate)
                
            # dfOriginal['entropy'] = np.array(entropyAllTrails).flatten()
            dfOriginal['crossEntropy'] = np.array(crossEntropyAllTrails).flatten()
            # dfOriginal['convergeRate'] =  np.array(convergeRateAllTrails).flatten()
            dfOriginal['Wolf Type'] = wolfType
            allDf.append(dfOriginal)
    
    measureLabelDict = {'entropy': 'Entropy of the Hunters\' Intention Distribution', 'crossEntropy': 'Cross Entropy of the Hunters\' Intention Distribution', 'convergeRate': 'Convergence of the Intentions of Hunters'}
    measure = 'crossEntropy'
    measureLabel = measureLabelDict[measure]
    
    dfTrialData = pd.concat(allDf)
    dfData = copy.deepcopy(dfTrialData)
    
    dfData['Stag Speed Multiplier'] = dfData['sheepMaxSpeed']
    dfData['Set Size'] = dfData['sheepNums']
    
    dfOutputStat = dfData[dfData['sheepNums'] > 1]
    dfOutputStat.reset_index()
    # dfOutputStat.set_index(['Stag Speed Multiplier', 'Set Size'], 'Wolf Type', 'Time Step'])

    fig = plt.figure()
    fig.set_dpi(120)
    numColumns = 2 # len of manipulatedVariables['Set Size']
    numRows = 3 # len of manipulatedVariables['Stag Speed Multiplier']
    plotCounter = 1
    print(dfOutputStat)
    for key, group in dfOutputStat.groupby(['Stag Speed Multiplier', 'Set Size']):

        group.reset_index(drop=True) 
        # group.index = group.index.droplevel(['Stag Speed Multiplier', 'Set Size'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if (plotCounter) % max(numColumns, 2) == 1:
           axForDraw.set_ylabel('Stag Speed Multiplier = '+str(key[0])) 
        else:
           axForDraw.set_ylabel(' ')
        if plotCounter <= numColumns:
           axForDraw.set_title('Set Size = '+str(key[1]))
        if plotCounter <= numColumns * (numRows - 1):
            axForDraw.set_xlabel(' ')
        print(plotCounter)
        grp = group.groupby(['name', 'Time Step', 'Wolf Type']).mean()
        # import ipdb; ipdb.set_trace()
        sns.lineplot(ax = axForDraw, data = grp, x = 'Time Step', y = measure, hue = 'Wolf Type', ci = 68)
        axForDraw.set_ylim(0, 10.8)
        plotCounter = plotCounter + 1


    plt.suptitle(measureLabel)
    #plt.legend(loc='best')
    #fig.text(x = 0.5, y = 0.04, s = 'Rationality In Inference', ha = 'center', va = 'center')
    #fig.text(x = 0.05, y = 0.5, s = 'valuePriorEndTime', ha = 'center', va = 'center', rotation=90)
    plt.show()

