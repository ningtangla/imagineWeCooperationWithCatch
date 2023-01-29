import os
import pickle
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import collections as co
import itertools as it
import functools as ft

import numpy as np
import random
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail, DrawImageWithJoysticksCheck, DrawNewStateWithBlocksAndFeedback
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trialCleaned import NewtonChaseTrialAllCondtionWithDiffBlocksWithDiffTargetColor, \
    NewtonChaseTrialAllCondtionPolicyPoolWithDiffBlocksWithDiffTargetColor, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experiment import NewtonExperimentWithDiffBlocks
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpWithNoise, GetCollisionForce,\
    ApplyActionForce, ApplyEnvironForce, getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState,\
    ObserveWithCaughtHistory, ReshapeActionVariousForce, ResetMultiAgentNewtonChasingVariousSheepWithCaughtHistoryWithDiffBlocks, \
    ResetStateWithCaughtHistory, CalSheepCaughtHistory, IntegrateStateWithCaughtHistory, RewardWolfWithBiteAndKill, \
    IsCollision, BuildGaussianFixCov, sampleFromContinuousSpace
from collections import OrderedDict


def main(subIndex):
    dirName = os.path.dirname(__file__)
    
    numTrainedSheepFolder = 5
    # numTrainedWolfFolder = 5
    numTrainedWolfFolderPostfix = [5]
    evaluateEpisodeWolf = 120000

    wolfControllerType = 'model'
    physicsDT = 0.05 # 50ms
    displayDT = int(physicsDT * 1000) # equal to physicsDT, *1000 since pygame use ms as unit
    trialCodeRunningTime = 3.37 # from code running time est for every timestep of forloop
    trialTime = 20 * 1000 # 30s
    maxTrialStep = int((trialTime - 600) / displayDT) # -600 from code running time test for initial running time before forloop

    wolfActionUpdateInterval = 'TempInterval'
    sheepActionUpdateInterval = 6
    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [999]
    manipulatedVariables['sheepWolfForceRatio'] = [1.0]
    manipulatedVariables['blockSize'] = [0]
    manipulatedVariables['targetColorIndex'] = [[0]]#, [0, 0], [0, 0, 0, 0]]#[[0], [1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
    manipulatedVariables['sheepConcern'] = ['self']
    # manipulatedVariables['sheepConcern'] = ['self', 'all']
    manipulatedVariables['sheepMaxSpeed'] = [0.7]#, 1.1, 1.5]
    trailNumEachCondition = 2
    trailNumEachConditionPractice = 0

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    allConditions = parametersAllCondtion * trailNumEachCondition
    # random.shuffle(allConditions)

    practiceConditions = parametersAllCondtion * trailNumEachConditionPractice
    random.shuffle(practiceConditions)
    
    rewardSharedOrIndividual = 'shared'
    experimentValues = co.OrderedDict()
    experimentValues["name"] = rewardSharedOrIndividual+ 'Reward' + str(evaluateEpisodeWolf) + 'wolfActionInterval' +str(wolfActionUpdateInterval) + '_sheepActionInterval' + str(sheepActionUpdateInterval) +'_'+str(subIndex) + '_' + str(numTrainedWolfFolderPostfix[0])

    resultsDicPath = os.path.join(dirName, '..', 'results')

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
                print('numSheeps', numSheeps)
                numAgents = numWolves + numSheeps
                numEntities = numAgents + numBlocks
                wolvesID = list(range(numWolves))
                sheepsID = list(range(numWolves, numAgents))
                blocksID = list(range(numAgents, numEntities))

                entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

                wolfMaxSpeed = 1.0
                # sheepMaxSpeed = 1.1
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
                reShapeAction = ReshapeActionVariousForce()
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
                transit = TransitMultiAgentChasingForExpWithNoise(reShapeAction, reShapeAction, applyActionForce,
                                                                  applyEnvironForce, integrateState, checkAllAgents,
                                                                  noiseAction)
                allTransitFun.update({(numSheeps, sheepMaxSpeed, blockSize): transit})

                # transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)

                def loadPolicyPoolOneCondition(numSheeps, sheepMaxSpeed, blockSize):
                    # -----------observe--------
                    if sheepConcern == 'self':
                        numSheepToObserve = 1
                    if sheepConcern == 'all':
                        numSheepToObserve = numSheeps

                    if blockSize > 0:
                        numBlocks = 2
                    else:
                        numBlocks = 0

                    wolvesIDForSheepObserve = list(range(numWolves))
                    sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
                    blocksIDForSheepObserve = list(
                        range(numSheeps + numWolves, numSheeps + numWolves + numBlocks))
                    observeOneAgentForSheep1 = lambda agentID, sId: ObserveWithCaughtHistory(agentID,
                                                                                             wolvesIDForSheepObserve,
                                                                                             sId,
                                                                                             blocksIDForSheepObserve,
                                                                                             getPosFromAgentState,
                                                                                             getVelFromAgentState,
                                                                                             getCaughtHistoryFromAgentState)
                    observeOneAgentForSheep = ft.partial(observeOneAgentForSheep1, sId=sheepsIDForSheepObserve)
                    observeOneForSheep = lambda state, num: [observeOneAgentForSheep(agentID)(state) for agentID in
                                                             range(num)]
                    sheepObserve = ft.partial(observeOneForSheep, num=numWolves + numSheepToObserve)
                    
                    sheepObsList = []
                    for sheepId in sheepsID:
                        obsFunList = [ObserveWithCaughtHistory(agentID, wolvesIDForSheepObserve, [sheepId],
                                                               blocksIDForSheepObserve,
                                                               getPosFromAgentState, getVelFromAgentState,
                                                               getCaughtHistoryFromAgentState)
                                      for agentID in list(range(numWolves)) + [sheepId]]
                        sheepObsLambda = lambda state, obsList: list([obs(state) for obs in obsList])
                        sheepObs = ft.partial(sheepObsLambda, obsList=obsFunList)
                        sheepObsList.append(sheepObs)
                        
                    initSheepObsForParams = sheepObserve(reset(numSheeps, blockSize))
                    obsSheepShape = [initSheepObsForParams[obsID].shape[0] for obsID in
                                     range(len(initSheepObsForParams))]

                    worldDim = 2
                    actionDim = worldDim * 2 + 1
                    layerWidth = [128, 128]

                    # -----------model--------
                    # modelFolderName = 'withoutWall3wolves'
                    # modelFolderName = 'withoutWall2wolves'
                    # modelFolderName = 'newRewardIndividualAllSheep2block12Wepisode'
                    # if blockSize == 0.26:
                    #     modelFolderName = '12Wepisode0.05dt1.0Mapsize0.26BlockSize1ForceRatio1SheepMaxSpeed'
                    #     maxEpisode = 120000
                    #     evaluateEpisode = 120000
                    # else:
                    #     modelFolderName = 'newRewardIndividalAllSheep2block8Wepisode0.05dt'
                    #     maxEpisode = 80000
                    #     evaluateEpisode = 80000
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
                    # sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0individ1.0_agent".format(numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                    # sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(
                    #     numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                    # sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}individ_agent".format(
                        # numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                    # sheepModelPathsAll = [
                        # os.path.join(modelFolderSheep, sheepFileNameAll + str(i) + str(evaluateEpisode) + 'eps') for i in
                        # range(numWolves, numWolves + numSheepToObserve)]
                    sheepModelPathsSep = [
                        os.path.join(modelFolderSheep,
                                     sheepFileNameSep + str(evaluateEpisode) + 'eps')
                        for modelFolderSheep in modelFoldersSheepListSep]
                    print(sheepModelPathsSep)
                    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

                    if sheepConcern == 'self':
                        [restoreVariables(model, path) for model, path in zip(sheepModelsListSep, sheepModelPathsSep)]
                        sheepModelsList = sheepModelsListSep
                        sheepPolicyFun = lambda allAgentsStates, models: list(
                            [actOneStepOneModel(model, sheepObsList[i](allAgentsStates)) for model,i in
                             zip(models, list(range(numSheeps)))])
                        sheepPolicyOneCondition = sheepPolicyFun
                    else:
                        [restoreVariables(model, path) for model, path in zip(sheepModelsListAll, sheepModelPathsAll)]
                        sheepModelsList = sheepModelsListAll
                        sheepPolicyFun = lambda allAgentsStates, models, obs: [actOneStepOneModel(model, obs(allAgentsStates))
                                                                       for model in models]
                        sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=sheepObserve)

                    if wolfControllerType == 'human':
                        humanControlPolicy = JoyStickForceControllers()
                        wolfPolicyOneCondition = lambda state, models: humanControlPolicy()
                        wolfModelsList = list(range(numWolves * numTrainedWolfFolder))
                    else:
                        observeOneAgent1 = lambda agentID, sId: ObserveWithCaughtHistory(agentID, wolvesID, sId, blocksID,
                                                                                   getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState)
                        # observeOneAgent1 = lambda agentID, sId: Observe(agentID, wolvesID, sId, blocksID, getPosFromAgentState,
                        #                                                 getVelFromAgentState)
                        observeOneAgent = ft.partial(observeOneAgent1, sId=sheepsID)
                        observeOne = lambda state, num: [observeOneAgent(agentID)(state) for agentID in range(num)]
                        observe = ft.partial(observeOne, num=numAgents)
                        initObsForParams = observe(reset(numSheeps, blockSize))
                        obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

                        buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
                        wolfModelsList = [ ]
                        for z in range(1 * len(numTrainedWolfFolderPostfix)):
                            wolfModelsList = wolfModelsList + [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves)] 

                        # modelFolderNameWolf = 'newRewardAllSheep0.2blockSize12Wepisode0.05dt10KillReward6SheepLife'
                        # modelFolderNameWolf = 'newRewardIndividalAllSheep2block8Wepisode0.05dt'
                        modelFolderNameWolf = 'newRewardAllSheep1.2killZoneRatio'
                        maxEpisodeWolf = 120000

                        maxTimeStepWolf = 75
                        modelSheepSpeedWolf1 = sheepMaxSpeed

                        modelFolderWolf = os.path.join(dirName, '..', 'model', modelFolderNameWolf)
                        modelFoldersWolfList = [os.path.join(dirName, '..', 'model', modelFolderNameWolf + str(j)) for j in list(numTrainedWolfFolderPostfix)]
                        print(modelFoldersWolfList)
                        wolfFileName = "maddpg{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheeps, numBlocks, maxEpisodeWolf)
                        wolfFileName1 = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}".format(numWolves, numSheeps, numBlocks, maxEpisodeWolf, maxTimeStepWolf, modelSheepSpeedWolf1) + rewardSharedOrIndividual + "_agent"
                        # wolfFileName1 = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves, numSheeps, numBlocks, maxEpisodeWolf, maxTimeStepWolf, modelSheepSpeedWolf1)
                        wolfModelPaths1 = [os.path.join(modelFolderWolf, wolfFileName1 + str(i) + str(evaluateEpisodeWolf) + 'eps') for (modelFolderWolf, i) in it.product(modelFoldersWolfList, list(range(numWolves)))]
                        modelSheepSpeedWolf2 = 1.3
                        wolfFileName2 = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves, numSheeps, numBlocks, maxEpisodeWolf, maxTimeStepWolf, modelSheepSpeedWolf2)
                        wolfModelPaths2 = [os.path.join(modelFolderWolf, wolfFileName2 + str(i) + str(evaluateEpisodeWolf) + 'eps') for (modelFolderWolf, i) in it.product(modelFoldersWolfList, list(range(numWolves)))]
                        
                        wolfModelPaths = wolfModelPaths1 #+ wolfModelPaths2

                        # wolfModelPathsSelected = [wolfModelPaths[x] for x in list(np.array(runningWolfIds[numSheeps]) - 1)]
                        [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
                        wolfPolicyFun = lambda allAgentsStates, models, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in models]
                        wolfPolicyOneCondition = ft.partial(wolfPolicyFun, obs=observe)
                    return sheepModelsList, sheepPolicyOneCondition, wolfModelsList, wolfPolicyOneCondition

                sheepModelsList, sheepPolicy, wolfModelsList, wolfPolicy = loadPolicyPoolOneCondition(numSheeps, sheepMaxSpeed, blockSize)
                    
                runningWolfIds = {1:[1, 2, 16, 17, 18, 4, 5, 19, 20, 21, 7, 8, 9, 22, 23, 24, 10, 11, 12, 25, 27, 13, 14, 15, 28, 29, 30],
                                                2:[2, 3, 16, 17, 18, 4, 6, 19, 20, 21, 8, 9, 22, 23, 24, 10, 11, 12, 25, 27, 14, 15, 28, 29],
                                                4:[2, 3, 18, 6, 19, 7, 8, 22, 12, 27, 13, 14, 15, 30]}
                allSheepModels.update({(numSheeps, sheepMaxSpeed, blockSize): sheepModelsList})
                allWolfModels.update({(numSheeps, sheepMaxSpeed, blockSize): wolfModelsList})
                # allWolfModels.update({(numSheeps, sheepMaxSpeed, blockSize): list(np.array(wolfModelsList)[np.array(runningWolfIds[numSheeps]) - 1])})
                allSheepPolicy.update({(numSheeps, sheepMaxSpeed, blockSize): sheepPolicy})
                allWolfPolicy.update({(numSheeps, sheepMaxSpeed, blockSize): wolfPolicy})
                
                screenWidth = int(800)
                screenHeight = int(800)
                fullScreen = False
                initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
                screen = initializeScreen()

                backgroundColor = THECOLORS['grey']  # [205, 255, 204]
                # targetColor = [THECOLORS['orange'], THECOLORS['chocolate1'], THECOLORS['tan1'], THECOLORS[
                #     'goldenrod2']]  # 'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)
                # targetColor = [THECOLORS['orange'], THECOLORS['purple']] * 8  # [255, 50, 50]
                # playerColors = [THECOLORS['red3'], THECOLORS['blue3'],THECOLORS['green4']]  # 'red3', (205, 0, 0); 'blue3', (0, 0, 205); 'green4', (0, 139, 0)
                targetColor = [THECOLORS['orange'], THECOLORS['blue3']] * 8  # [255, 50, 50]
                playerColors = [THECOLORS['red3'], THECOLORS['red3'],THECOLORS['red3']]  # 'red3', (205, 0, 0); 'blue3', (0, 0, 205); 'green4', (0, 139, 0)
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

    killzone = (wolfSize + sheepSize) * killZoneRatio
    recordEaten = RecordEatenNumber(isAnyKilled)
    allWolfActionUpdateInterval = {0.7: 14, 1.1: 11, 1.5: 13}
    trial = NewtonChaseTrialAllCondtionPolicyPoolWithDiffBlocksWithDiffTargetColor(screen, killzone, sheepLife, targetColor, numWolves, numBlocks,
                 maxTrialStep, allWolfActionUpdateInterval, sheepActionUpdateInterval, displayDT, allSheepModels, allWolfModels, 
                 stopwatchEvent, allDrawNewStateFun, getEntityPos, getEntityVel, getEntityCaughtHistory, 
                 recordEaten, allWolfPolicy, allSheepPolicy, allTransitFun, allWolfRewardFun)
                 
    print('aaa')
    hasRest = True
    # resetWithCaughtHistory = ResetStateWithCaughtHistory(reset, calSheepCaughtHistory)

    picturePath = os.path.abspath(os.path.join(os.path.join(dirName, '..'), 'pictures'))
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction-waitall-color.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest-waitall.png'))
    practiceFinishImage = pg.image.load(os.path.join(picturePath, 'practiceFinish.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    drawImage = DrawImage(screen)
    if wolfControllerType == 'human':
        humanControlPolicy = JoyStickForceControllers()
        drawStartAndRestImgExp = DrawImageWithJoysticksCheck(screen, humanControlPolicy.joystickList)
    if wolfControllerType == 'model':
        drawStartAndRestImgExp = drawImage
    experimentPractice = NewtonExperimentWithDiffBlocks(restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset,
                                  drawStartAndRestImgExp)
    experiment = NewtonExperimentWithDiffBlocks(restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset,
                                  drawStartAndRestImgExp, writable = True)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    # drawStartAndRestImgExp(introductionImage)
    restTimesPractice = 1
    if trailNumEachConditionPractice > 0:
        experimentPractice(trialTime, practiceConditions, restTimesPractice)
        drawStartAndRestImgExp(practiceFinishImage)

    block = 1
    restTimes = 1  # the number of breaks in an experiment
    for i in range(block):
        experiment(trialTime, allConditions, restTimes)
        # giveExperimentFeedback(i, score)
        # if i == block - 1:
            # drawImage(finishImage)
        # else:
            # drawStartAndRestImgExp(restImage)


if __name__ == "__main__":
    subNumTotal = 16
    for subIndex in range(subNumTotal):#, subNumTotal):
        main(subIndex)