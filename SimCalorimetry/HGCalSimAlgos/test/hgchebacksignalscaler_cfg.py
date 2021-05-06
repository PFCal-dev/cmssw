#run with: cmsRun hgchebacksignalscaler_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.5.15.9.txt sipmMap=SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt nPEperMIP=21

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register ("doseMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("sipmMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("nPEperMIP", "",  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("pxFiringRate","",  VarParsing.multiplicity.singleton, VarParsing.varType.float)
options.parseArguments()

process = cms.Process("demo",eras.Phase2C9)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.plotter = cms.EDAnalyzer("HGCHEbackSignalScalerAnalyzer",
                                 doseMap  = cms.string( options.doseMap ),
                                 sipmMap  = cms.string( options.sipmMap ),
                                 pxFiringRate=cms.double(options.pxFiringRate),                                 
                                 nPEperMIP = cms.uint32( options.nPEperMIP )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("dosemap_output_sci.root")
)

process.p = cms.Path(process.plotter)
