import FWCore.ParameterSet.Config as cms
import math


def custom_towers_etaphi(process, 
        minEta=1.479,
        maxEta=3.0,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72,
        binsEta=[],
        binsPhi=[]
        ):
    parameters_towers_2d = process.hgcalTowerMapProducer.ProcessorParameters.towermap_parameters
    parameters_towers_2d.L1TTriggerTowerConfig.readMappingFile = cms.bool(False)
    parameters_towers_2d.L1TTriggerTowerConfig.minEta = cms.double(minEta)
    parameters_towers_2d.L1TTriggerTowerConfig.maxEta = cms.double(maxEta)
    parameters_towers_2d.L1TTriggerTowerConfig.minPhi = cms.double(minPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.maxPhi = cms.double(maxPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsEta = cms.int32(nBinsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsPhi = cms.int32(nBinsPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.binsEta = cms.vdouble(binsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.binsPhi = cms.vdouble(binsPhi)
    return process


def custom_towers_map(process, 
        towermapping='L1Trigger/L1THGCal/data/tower_mapping_hgcroc_eta-phi_v0.txt',
        minEta=1.41,
        maxEta=3.1,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72
        ):
    parameters_towers_2d = process.hgcalTowerMapProducer.ProcessorParameters.towermap_parameters
    parameters_towers_2d.L1TTriggerTowerConfig.readMappingFile = cms.bool(True)
    parameters_towers_2d.L1TTriggerTowerConfig.L1TTriggerTowerMapping = cms.FileInPath(towermapping)
    parameters_towers_2d.L1TTriggerTowerConfig.minEta = cms.double(minEta)
    parameters_towers_2d.L1TTriggerTowerConfig.maxEta = cms.double(maxEta)
    parameters_towers_2d.L1TTriggerTowerConfig.minPhi = cms.double(minPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.maxPhi = cms.double(maxPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsEta = cms.int32(nBinsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsPhi = cms.int32(nBinsPhi)
    return process
