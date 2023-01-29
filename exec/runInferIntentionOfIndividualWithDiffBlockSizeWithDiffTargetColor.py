import os
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import collections as co
import itertools as it
import functools as ft
from collections import OrderedDict
import pandas as pd
import copy

import numpy as np
import random
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewStateWithBlocks,  DrawNewStateWithBlocksAndFeedback, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail, DrawImageWithJoysticksCheck
from src.writer import WriteDataFrameToCSV
from src.trialCleanedInferIntentionOfIW import NewtonChaseTrialInference, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experimentHybridTeam import NewtonExperimentWithResetIntentionHybridTeam
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy, actByPolicyTrainNoNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
from src.mathTools.distribution import sampleFromDistribution,  SoftDistribution, BuildGaussianFixCov, sampleFromContinuousSpace
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpWithNoise, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, IsCollision, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState, ObserveWithCaughtHistory, ReshapeWolfAction, ReshapeActionVariousForce, ResetMultiAgentNewtonChasingVariousSheep, \
    ResetStateWithCaughtHistory, CalSheepCaughtHistory, IntegrateStateWithCaughtHistory, RewardWolfWithBiteAndKill, \
    BuildGaussianFixCov, sampleFromContinuousSpace, ComposeCentralControlPolicyByGaussianOnDeterministicAction, ResetMultiAgentNewtonChasingVariousSheepWithCaughtHistoryWithDiffBlocks
from src.MDPChasing.policy import RandomPolicy
from src.inference.intention import UpdateIntention
from src.inference.percept import SampleNoisyAction, PerceptImaginedWeAction
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsContinuousPolicyLikelihood, InferOneStep
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, SampleIndividualActionGivenIntention, SampleActionOnChangableIntention
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.writer import loadFromPickle, saveToPickle


class CalJointLikelihood:
    def __init__(self, agentID, calCommittedAgentsPolicyLikelihood, calUncommittedAgentsPolicyLikelihood):
        self.agentID = agentID
        self.calCommittedAgentsPolicyLikelihood = calCommittedAgentsPolicyLikelihood
        self.calUncommittedAgentsPolicyLikelihood = calUncommittedAgentsPolicyLikelihood
    def __call__(self, intention, state, perceivedAction):
        return self.calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
                    self.calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)
                    
def main():
    dirName = os.path.dirname(__file__)
    fileFolder = os.path.join(dirName, '..', 'results', 'Expt 3 Data')
    prefixList = []
    allFiles = os.listdir(fileFolder)
    for j in allFiles:
        if os.path.splitext(j)[1] == '.pickle':
            prefixList.append(os.path.splitext(j)[0])
    
    print(prefixList)
    for prefix in prefixList:
        fileName = os.path.join(dirName, '..', 'results', 'Expt 3 Data', prefix + '.pickle')
        humanTrailsRawData = loadFromPickle(fileName)
    
        numTrainedSheepFolder = 5
        # numTrainedWolfFolder = 5
        numTrainedWolfFolderPostfix = [1]
        evaluateEpisodeWolf = 120000
        
        wolfControllerType = 'model'
        physicsDT = 0.05 # 50ms
        displayDT = int(physicsDT * 1000) # equal to physicsDT, *1000 since pygame use ms as unit
        trialCodeRunningTime = 3.37 # from code running time est for every timestep of forloop
        trialTime = 20 * 1000 # 30s
        maxTrialStep = int((trialTime - 600) / displayDT) # -600 from code running time test for initial running time before forloop

        wolfActionUpdateInterval = 11
        sheepActionUpdateInterval = 6
        manipulatedVariables = OrderedDict()
        manipulatedVariables['sheepNums'] = [999]
        manipulatedVariables['sheepWolfForceRatio'] = [1.0]
        manipulatedVariables['blockSize'] = [0]
        manipulatedVariables['targetColorIndex'] = [[0], [0, 0], [0, 0, 0, 0]]#[[0], [1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        manipulatedVariables['sheepConcern'] = ['self']
        # manipulatedVariables['sheepConcern'] = ['self', 'all']
        manipulatedVariables['sheepMaxSpeed'] = [0.7, 1.1, 1.5]
        priorDecayRate = 0.6
        deviationFor2DAction = 0.5
        rationalityBetaInInference = 0.95
        valuePriorEndTime = -100
        trailNumEachCondition = 13
        trailNumEachConditionPractice = 0

        productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
        parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
        allConditions = parametersAllCondtion * trailNumEachCondition
        # random.shuffle(allConditions)

        practiceConditions = parametersAllCondtion * trailNumEachConditionPractice
        random.shuffle(practiceConditions)

        resultsDicPath = os.path.join(dirName, '..', 'results')

        rewardSharedOrIndividual = 'shared'
        experimentValues = co.OrderedDict()
        experimentValues["name"] = prefix + '_deviationFor2DAction_' + str(deviationFor2DAction) \
                                                    + '_priorDecayRate_' + str(priorDecayRate) + '_betaInInference_' + str(rationalityBetaInInference)

        # experimentValues["name"] = '0704'
        writerPath = os.path.join(resultsDicPath, experimentValues["name"]) + '.csv'
        picklePath = os.path.join(resultsDicPath, experimentValues["name"]) + '.pickle'
        writer = WriteDataFrameToCSV(writerPath)
        pickleWriter = lambda data: saveToPickle(data, picklePath)
        
        # --------environment setting-----------
        mapSize = 1.0
        displaySize = 1.0
        minDistance = mapSize * 1 / 3
        minDistanceInitBlocks = max(manipulatedVariables['blockSize']) * 1.5
        wolfSize = 0.065
        sheepSize = 0.065
        killZoneRatio = 1.2
                    
        numWolves = 3
        experimentValues["numWolves"] = numWolves
        sheepLife = 6 #20 for terminalGame
        biteReward = 0.1 #0 for terminalGame
        killReward = 1 #0 for terminalGame
        sheepConcern = 'self'
        
        allGetIntentionDistributions =  {}
        allRecordActionForUpdateIntention =  {}
        allResetIntenions =  {}

        allSheepModels = {}
        allWolfModels = {}
        allWolfPolicy = {}
        allSheepPolicy = {}
        allWolfRewardFun = {}
        allTransitFun = {}
        allDrawNewStateFun = {}
        for blockSize in manipulatedVariables['blockSize']:
            for targetColorIndex in manipulatedVariables['targetColorIndex']:
                for sheepMaxSpeed in manipulatedVariables['sheepMaxSpeed']:
                
                    if blockSize > 0:
                        numBlocks = 2
                    else:
                        numBlocks = 0
                    numSheeps = len(targetColorIndex)
                    # print('numSheeps', numSheeps)
                    numAgents = numWolves + numSheeps
                    numEntities = numAgents + numBlocks
                    wolvesID = list(range(numWolves))
                    sheepsID = list(range(numWolves, numAgents))
                    blocksID = list(range(numAgents, numEntities))
                    possibleWolvesIds = wolvesID
                    possibleSheepIds = sheepsID

                    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
                    
                    wolfMaxSpeed = 1
                    blockMaxSpeed = None

                    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
                    entitiesMovableList = [True] * numAgents + [False] * numBlocks
                    massList = [1.0] * numEntities
                    reset = ResetMultiAgentNewtonChasingVariousSheepWithCaughtHistoryWithDiffBlocks(numWolves, numBlocks, mapSize, minDistance, minDistanceInitBlocks)
                    isCollision = IsCollision(getPosFromAgentState, killZoneRatio)
                    rewardWolf = RewardWolfWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, isCollision, getCaughtHistoryFromAgentState, sheepLife, biteReward, killReward)
                    allWolfRewardFun.update({(numSheeps, sheepMaxSpeed): rewardWolf})
                    
                    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-displaySize, displaySize], [-displaySize, displaySize])
                    
                    def checkBoudary(agentState):
                        newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
                        return newState

                    def checkBoudaryWithCaughtHistory(agentState):
                        position = getPosFromAgentState(agentState)
                        velocity = getVelFromAgentState(agentState)
                        adjustedState = stayInBoundaryByReflectVelocity(position, velocity)
                        if len(agentState) == 5:
                            adjustedState = np.append(adjustedState, getCaughtHistoryFromAgentState(agentState))
                        return adjustedState.copy()

                    # checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
                    checkAllAgents = lambda states: [checkBoudaryWithCaughtHistory(agentState) for agentState in states]
                    # reshapeHumanAction = ReshapeHumanAction()
                    # reshapeSheepAction = ReshapeSheepAction()
                    reshapeWolfAction = lambda action, force: action
                    reshapeSheepAction = ReshapeActionVariousForce()
                    getCollisionForce = GetCollisionForce()
                    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
                    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                                          getPosFromAgentState)
                    calSheepCaughtHistory = CalSheepCaughtHistory(wolvesID, numBlocks, entitiesSizeList, isCollision, sheepLife)
                    integrateState = IntegrateStateWithCaughtHistory(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                                    getVelFromAgentState, getPosFromAgentState, calSheepCaughtHistory, damping=0.25, dt=physicsDT)

                    actionDimReshaped = 2
                    cov = [0.3 ** 2 for _ in range(actionDimReshaped)]
                    buildGaussian = BuildGaussianFixCov(cov)
                    noiseAction = lambda action: sampleFromContinuousSpace(buildGaussian(tuple(action)))
                    transit = TransitMultiAgentChasingForExpWithNoise(reshapeWolfAction, reshapeSheepAction, applyActionForce,
                                                                      applyEnvironForce, integrateState, checkAllAgents,
                                                                      noiseAction)
                    allTransitFun.update({(numSheeps, sheepMaxSpeed, blockSize): transit})

                    # Inference Part (Individual Inference)
                    numWolvesInInference = 1
                    updateIntentions = []
                    wolvesIntentionPriors = []
                    for wolfIdInInference in range(numWolves):
                        intentionSpaces = tuple(it.product(possibleSheepIds, [tuple([wolfIdInInference])]))
                        wolvesIntentionPrior = {tuple(intention): 1 / len(intentionSpaces) for intention in intentionSpaces}
                        wolvesIntentionPriors.append(wolvesIntentionPrior)
                        # Percept Action For Inference
                        # perceptAction = lambda action: action
                        perceptSelfAction = SampleNoisyAction(deviationFor2DAction)
                        perceptOtherAction = SampleNoisyAction(deviationFor2DAction)
                        perceptAction = PerceptImaginedWeAction(possibleWolvesIds, perceptSelfAction, perceptOtherAction)

                        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
                        # ------------ wolf model -------------
                        weModelsListBaseOnNumInWe = []
                        observeListBaseOnNumInWe = []
                        for numAgentInWe in range(numWolvesInInference, numWolvesInInference + 1):
                            worldDim = 2
                            actionDim = worldDim * 2 + 1
                            numSheepForWe = 1
                            numBlocksForWe = numBlocks
                            wolvesIDForWolfObserve = list(range(numAgentInWe))
                            sheepsIDForWolfObserve = list(range(numAgentInWe, 1 + numAgentInWe))
                            blocksIDForWolfObserve = list(range(1 + numAgentInWe, 1 + numAgentInWe + numBlocksForWe))
                            observeOneAgentForWolf = lambda agentID: ObserveWithCaughtHistory(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve,
                                                                             blocksIDForWolfObserve, getPosFromAgentState,
                                                                             getVelFromAgentState, getCaughtHistoryFromAgentState)
                            observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numAgentInWe + 1)]
                            observeListBaseOnNumInWe.append(observeWolf)

                            obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
                            initObsForWolfParams = observeWolf(reset(numSheepForWe, blockSize)[obsIDsForWolf])
                            obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
                            buildWolfModels = BuildMADDPGModels(actionDim, numAgentInWe + 1, obsShapeWolf)
                            # layerWidthForWolf = [64 * (numAgentInWe - 1), 64 * (numAgentInWe - 1)]
                            layerWidthForWolf = [128, 128]
                            wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numAgentInWe)]

                            modelFolderNameWolf = 'oneWolfOneSheep1.2killZoneRatio' + str(numTrainedWolfFolderPostfix[0])
                            maxEpisodeWolf = 120000

                            maxTimeStepWolf = 75
                            modelSheepSpeedWolf1 = 1.0 #sheepMaxSpeed
                            wolfFileName = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}".format(numWolvesInInference,
                                                                                                                     numBlocks,
                                                                                                                     maxEpisodeWolf,
                                                                                                                     maxTimeStepWolf,
                                                                                                                     modelSheepSpeedWolf1) + rewardSharedOrIndividual + "_agent"
                            wolfModelPaths = [os.path.join(dirName, '..', 'model', modelFolderNameWolf, wolfFileName + str(i) + str(evaluateEpisodeWolf) + 'eps') for i in
                                              range(numWolvesInInference)]
                            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
                            weModelsListBaseOnNumInWe.append(wolfModelsList)
                            # import ipdb; ipdb.set_trace()
                            # print('loadModel', len(weModelsListBaseOnNumInWe))

                        # For action inference
                        actionDimReshaped = 2
                        cov = [deviationFor2DAction ** 2 for _ in range(actionDimReshaped)]
                        buildGaussian = BuildGaussianFixCov(cov)
                        actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoNoisy)
                        # actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoisy)
                        reshapeAction = ReshapeWolfAction()
                        composeCentralControlPolicy = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction,
                            observe, actOneStepOneModelWolf, buildGaussian)
                        wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 1])(
                            weModelsListBaseOnNumInWe[numAgentInWe - 1], numAgentsInWe) for numAgentsInWe in range(numWolvesInInference, numWolvesInInference + 1)]
                        # wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 2])(weModelsListBaseOnNumInWe[numAgentsInWe - 2], numAgentsInWe)
                        # for numAgentsInWe in range(2, numWolves + 1)]
                        centralControlPolicyListBasedOnNumAgentsInWe = wolvesCentralControlPolicies  
                        numOfAgentInPolicyFunctionList = [1]
                        softPolicyInInference = lambda distribution: distribution
                        getStateThirdPersonPerspective = lambda state, goalId, weIds: getStateOrActionThirdPersonPerspective(state,
                                                                                                                             goalId,
                                                                                                                             weIds,
                                                                                                                             blocksID)
                        policyForCommittedAgentsInInference = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe,
                                                                                      softPolicyInInference,
                                                                                      getStateThirdPersonPerspective, numOfAgentInPolicyFunctionList)
                        concernedAgentsIds = [wolfIdInInference]
                        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsContinuousPolicyLikelihood(concernedAgentsIds,
                                                                                                          policyForCommittedAgentsInInference,
                                                                                                          rationalityBetaInInference)

                        randomActionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5), (0, 0)]
                        randomPolicy = RandomPolicy(randomActionSpace)
                        getStateFirstPersonPerspective = lambda state, goalId, weIds, selfId: getStateOrActionFirstPersonPerspective(
                            state, goalId, weIds, selfId, blocksID)
                        allAgentsIdsWholeGroup = [wolfIdInInference]
                        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(allAgentsIdsWholeGroup, randomPolicy,
                                                                                          softPolicyInInference,
                                                                                          getStateFirstPersonPerspective)
                        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(allAgentsIdsWholeGroup,
                                                                                                    concernedAgentsIds,
                                                                                                    policyForUncommittedAgentsInInference)
                        # Joint Likelihood
                        # calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
                                                                                       # calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)
                        
                        calJointLikelihood = CalJointLikelihood(wolfIdInInference, calCommittedAgentsPolicyLikelihood, calUncommittedAgentsPolicyLikelihood)
                        
                        # Infer and update Intention
                        variables = copy.deepcopy([intentionSpaces])
                        jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention']) 
                        concernedHypothesisVariable = ['intention']
                        # priorDecayRate = 1
                        softPrior = SoftDistribution(priorDecayRate)
                        inferIntentionOneStep = InferOneStep(jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood, softPrior)
                        # print(inferIntentionOneStep.calJointLikelihood.calUncommittedAgentsPolicyLikelihood.concernedAgentsIds)
                        if numSheeps == 1:
                            inferIntentionOneStep = lambda prior, state, action: prior

                        adjustIntentionPriorGivenValueOfState = lambda state: 1
                        chooseIntention = sampleFromDistribution
                        updateIntention = UpdateIntention(wolvesIntentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState,
                                                            perceptAction, inferIntentionOneStep, chooseIntention)
                        updateIntentions.append(updateIntention)

                    # Wolves Generate Action
                    covForPlanning = [0.03 ** 2 for _ in range(actionDimReshaped)]
                    buildGaussianForPlanning = BuildGaussianFixCov(covForPlanning)
                    composeCentralControlPolicyForPlanning = lambda \
                        observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction \
                        (reshapeAction, observe, actOneStepOneModelWolf, buildGaussianForPlanning)
                    wolvesCentralControlPoliciesForPlanning = [composeCentralControlPolicyForPlanning(
                        observeListBaseOnNumInWe[numAgentsInWe - 1])(weModelsListBaseOnNumInWe[numAgentsInWe - 1], numAgentsInWe)
                                                               for numAgentsInWe in range(numWolvesInInference, numWolvesInInference + 1)]

                    centralControlPolicyListBasedOnNumAgentsInWeForPlanning = wolvesCentralControlPoliciesForPlanning  # 0 for one agents in We, no two or three agents in We...
                    softPolicyInPlanning = lambda distribution: distribution
                    policyForCommittedAgentInPlanning = PolicyForCommittedAgent(
                        centralControlPolicyListBasedOnNumAgentsInWeForPlanning, softPolicyInPlanning,
                        getStateThirdPersonPerspective, numOfAgentInPolicyFunctionList)

                    policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy,
                                                                                    softPolicyInPlanning,
                                                                                    getStateFirstPersonPerspective)

                    def wolfChooseActionMethod(individualContinuousDistributions):
                        centralControlAction = tuple([tuple(sampleFromContinuousSpace(distribution))
                                                      for distribution in individualContinuousDistributions])
                        return centralControlAction


                    getSelfActionThirdPersonPerspective = lambda weIds, selfId: list(weIds).index(selfId)
                    chooseCommittedAction = GetActionFromJointActionDistribution(wolfChooseActionMethod,
                                                                                 getSelfActionThirdPersonPerspective)
                    chooseUncommittedAction = sampleFromDistribution
                    wolvesSampleIndividualActionGivenIntentionList = [
                        SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentInPlanning,
                                                             policyForUncommittedAgentInPlanning, chooseCommittedAction,
                                                             chooseUncommittedAction) for selfId in possibleWolvesIds]
                    wolvesSampleActions = [SampleActionOnChangableIntention(updateIntention, wolvesSampleIndividualActionGivenIntention)
                        for updateIntention, wolvesSampleIndividualActionGivenIntention in
                        zip(updateIntentions, wolvesSampleIndividualActionGivenIntentionList)]
                    
                    allWolfPolicy.update({(numSheeps, sheepMaxSpeed, blockSize): wolvesSampleActions})
                    # print(wolvesSampleActions[0].updateIntention.intentionPrior)
                    # print(wolvesSampleActions[1].updateIntention.intentionPrior)
                    # print(wolvesSampleActions[2].updateIntention.intentionPrior)
                    # reset intention and adjuste intention prior attributes tools for multiple trajectory
                    intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
                    intentionResetAttributeValues = [
                        dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
                        for intentionPrior in wolvesIntentionPriors]
                    resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)
                    returnAttributes = ['formerIntentionPriors']
                    getIntentionDistributions = GetObjectsValuesOfAttributes(returnAttributes, updateIntentions)
                    attributesToRecord = ['lastAction']
                    recordActionForUpdateIntention = RecordValuesForObjects(attributesToRecord, updateIntentions)
                    allGetIntentionDistributions.update({(numSheeps, sheepMaxSpeed, blockSize): getIntentionDistributions})
                    allRecordActionForUpdateIntention.update({(numSheeps, sheepMaxSpeed, blockSize): recordActionForUpdateIntention})
                    allResetIntenions.update({(numSheeps, sheepMaxSpeed, blockSize): resetIntentions})
                    
                    
                    # ------------ sheep model -------------
                    def loadPolicyOneCondition(numSheeps, sheepMaxSpeed, blockSize):
                        # -----------observe--------
                        if sheepConcern == 'self':
                            numSheepToObserve = 1
                        if sheepConcern == 'all':
                            numSheepToObserve = numSheeps

                        wolvesIDForSheepObserve = list(range(numWolves))
                        sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
                        blocksIDForSheepObserve = list(range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
                        observeOneAgentForSheep1 = lambda agentID, sId: ObserveWithCaughtHistory(agentID, wolvesIDForSheepObserve, sId, blocksIDForSheepObserve,
                                                                                getPosFromAgentState, getVelFromAgentState,  getCaughtHistoryFromAgentState)
                        observeOneAgentForSheep = ft.partial(observeOneAgentForSheep1, sId=sheepsIDForSheepObserve)
                        observeOneForSheep = lambda state, num: [observeOneAgentForSheep(agentID)(state) for agentID in
                                                                 range(num)]
                        sheepObserve = ft.partial(observeOneForSheep, num=numWolves + numSheepToObserve)
                        sheepObsList = []
                        for sheepId in sheepsID:
                            obsFunList = [ObserveWithCaughtHistory(agentID, wolvesIDForSheepObserve, [sheepId], blocksIDForSheepObserve,
                                                  getPosFromAgentState, getVelFromAgentState,  getCaughtHistoryFromAgentState) for agentID in list(range(numWolves)) + [sheepId]]
                            sheepObsLambda = lambda state, obsList: list([obs(state) for obs in obsList])
                            sheepObs = ft.partial(sheepObsLambda, obsList=obsFunList)
                            sheepObsList.append(sheepObs)
                        initSheepObsForParams = sheepObserve(reset(numSheepToObserve, blockSize))
                        obsSheepShape = [initSheepObsForParams[obsID].shape[0] for obsID in range(len(initSheepObsForParams))]

                        worldDim = 2
                        actionDim = worldDim * 2 + 1
                        layerWidth = [128, 128]

                        # -----------restore model--------
                        modelFolderNameSheep = 'newRewardIndividalAllSheep2block8Wepisode0.05dt'
                        maxEpisode = 80000
                        evaluateEpisode = 80000
                        maxTimeStep = 75
                        modelSheepSpeed = 1.0

                        buildSheepMADDPGModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsSheepShape)
                        # sheepModelsListAll = [buildSheepMADDPGModels(layerWidth, agentID) for agentID in
                                              # range(numWolves, numWolves + numSheepToObserve)]
                        sheepModelsListSep = [buildSheepMADDPGModels(layerWidth, agentID) for agentID in
                                              range(numWolves, numWolves + numSheepToObserve) for i in range(numTrainedSheepFolder)]

                        modelFolderSheep = os.path.join(dirName, '..', 'model', modelFolderNameSheep)
                        modelFoldersSheepListSep = [os.path.join(dirName, '..', 'model', modelFolderNameSheep + str(i+1)) for i in range(numTrainedSheepFolder)]
                        sheepFileNameSep = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}individ_agent3".format(
                            numWolves, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                        # sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(
                        #     numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                        # sheepModelPathsAll = [
                            # os.path.join(modelFolderSheep, sheepFileNameAll + str(i) + str(evaluateEpisode) + 'eps') for i in
                            # range(numWolves, numWolves + numSheepToObserve)]
                        sheepModelPathsSep = [
                            os.path.join(modelFolderSheep,
                                         sheepFileNameSep + str(evaluateEpisode) + 'eps')
                            for modelFolderSheep in modelFoldersSheepListSep]
                        print(sheepModelPathsSep)
                        actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

                        if numSheepToObserve == 1:
                            [restoreVariables(model, path) for model, path in zip(sheepModelsListSep, sheepModelPathsSep)]
                            # print(len(sheepModelsListSep), len(sheepObsList))
                            sheepPolicyFun = lambda allAgentsStates: list([actOneStepOneModel(model, sheepObsList[i](allAgentsStates)) for model,i in
                                             zip(sheepModelsListSep, list(range(numSheeps)))])
                            sheepPolicyOneCondition = sheepPolicyFun
                        else:
                            [restoreVariables(model, path) for model, path in zip(sheepModelsListAll, sheepModelPathsAll)]
                            sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in sheepModelsListAll]
                            sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=sheepObserve)
                        return sheepPolicyOneCondition

                    sheepPolicy = loadPolicyOneCondition(numSheeps, sheepMaxSpeed, blockSize)
                    allSheepPolicy.update({(numSheeps, sheepMaxSpeed, blockSize): sheepPolicy})

                    # transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)
                    
                    screenWidth = int(800)
                    screenHeight = int(800)
                    fullScreen = False
                    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
                    screen = initializeScreen()
                    
                    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
                    targetColor = [THECOLORS['orange'], THECOLORS['chocolate1'], THECOLORS['tan1'], THECOLORS['goldenrod2']]
                    #'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)
                    # targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
                    playerColors = [THECOLORS['red3'], THECOLORS['blue3'], THECOLORS['green4']]
                    blockColors = [THECOLORS['white']] * 2
                    textColorTuple = THECOLORS['green']
                    
                    gridSize = 40
                    leaveEdgeSpace = 5
                    playerRadius = int(wolfSize/(displaySize * 2)*screenWidth*gridSize/(gridSize+ 2 * leaveEdgeSpace))
                    targetRadius = int(sheepSize/(displaySize * 2)*screenWidth*gridSize/(gridSize+ 2 * leaveEdgeSpace))
                    blockRadius = int(blockSize/(displaySize * 2)*screenWidth*gridSize/(gridSize+2 * leaveEdgeSpace))
                    blockRadius = int(blockSize / (displaySize * 2) * screenWidth * gridSize / (gridSize + 2 * leaveEdgeSpace))
                    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
                    drawNewState = DrawNewStateWithBlocksAndFeedback(screen, drawBackground, playerColors, blockColors, targetRadius,
                                                                     playerRadius, blockRadius, displaySize, sheepLife)

                    allDrawNewStateFun.update({(numSheeps, sheepMaxSpeed, blockSize): drawNewState})

        stopwatchUnit = 100

        stopwatchEvent = pg.USEREVENT + 1
        pg.time.set_timer(stopwatchEvent, stopwatchUnit)
        pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
        pg.key.set_repeat(120, 120)

        getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
        getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
        getEntityCaughtHistory = lambda state, entityID: getCaughtHistoryFromAgentState(state[entityID])
        
        # checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
        killzone = (wolfSize + sheepSize) * killZoneRatio
        recordEaten = RecordEatenNumber(isAnyKilled)
        # humanController = JoyStickForceControllers()
        reshapeWolfActionInHybrid = ReshapeWolfAction()
        # reshapeWolfActionInHybrid = lambda action: action
        trial = NewtonChaseTrialInference(screen, killzone, sheepLife, targetColor, numWolves, numBlocks, stopwatchEvent, maxTrialStep, 
                                                               wolfActionUpdateInterval, sheepActionUpdateInterval, displayDT,
                                                               allDrawNewStateFun, recordEaten,  getEntityPos, getEntityVel,  getEntityCaughtHistory, allWolfPolicy, allSheepPolicy, allTransitFun, 
                                                               allWolfRewardFun,  allGetIntentionDistributions, allRecordActionForUpdateIntention, allResetIntenions, reshapeWolfActionInHybrid)

        picturePath = os.path.abspath(os.path.join(os.path.join(dirName, '..'), 'pictures'))
        # introductionImage = pg.image.load(os.path.join(picturePath, 'introduction-waitall.png'))
        restImage = pg.image.load(os.path.join(picturePath, 'rest-waitall.png'))
        finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
        # introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
        drawImage = DrawImage(screen)
        hasRest = False  # True
        experiment = NewtonExperimentWithResetIntentionHybridTeam(restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset,
                                      drawImage, writable = True)
        # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
        # drawImageBoth(introductionImage)
        block = 1
        restTimes = 3  # the number of breaks in an experiment
        for i in range(block):
            experiment(humanTrailsRawData, restTimes)
            # giveExperimentFeedback(i, score)
            # if i == block - 1:
                # drawImage(finishImage)
            # else:
                # drawImage(restImage)



if __name__ == "__main__":
    main()
