# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step2 --conditions auto:phase2_realistic -n 10 --era Phase2C8 --eventcontent FEVTDEBUGHLT -s DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,HLT:@fake2 --datatier GEN-SIM --beamspot NoSmear --geometry Extended2023D41 --pileup AVE_200_BX_25ns --pileup_input das:/RelValMinBias_14TeV/CMSSW_10_4_0_mtd3-103X_upgrade2023_realistic_v2_2023D35noPU_2-v1/GEN-SIM --no_exec --python_filename=D_Pileup_fragment.py
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8

#parse command line arguments
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('standard')
options.register('input', 'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/Mu50Gun/Events_338.root', VarParsing.multiplicity.singleton, VarParsing.varType.string, "input file to digitize")
options.register('minbias', 'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias', VarParsing.multiplicity.singleton, VarParsing.varType.string, "input file to digitize")
options.register('pileup', 140, VarParsing.multiplicity.singleton, VarParsing.varType.int, "average pileup")
options.parseArguments()

process = cms.Process('HLT',Phase2C8)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_Fake2_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("PoolSource",
                            dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
                            fileNames = cms.untracked.vstring(options.input.split(',')),
                            inputCommands = cms.untracked.vstring(
                                'keep *',
                                'drop *_genParticles_*_*',
                                'drop *_genParticlesForJets_*_*',
                                'drop *_kt4GenJets_*_*',
                                'drop *_kt6GenJets_*_*',
                                'drop *_iterativeCone5GenJets_*_*',
                                'drop *_ak4GenJets_*_*',
                                'drop *_ak7GenJets_*_*',
                                'drop *_ak8GenJets_*_*',
                                'drop *_ak4GenJetsNoNu_*_*',
                                'drop *_ak8GenJetsNoNu_*_*',
                                'drop *_genCandidatesForMET_*_*',
                                'drop *_genParticlesForMETAllVisible_*_*',
                                'drop *_genMetCalo_*_*',
                                'drop *_genMetCaloAndNonPrompt_*_*',
                                'drop *_genMetTrue_*_*',
                                'drop *_genMetIC5GenJs_*_*'
                            ),
                            secondaryFileNames = cms.untracked.vstring()
                        )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('Events_'+str(options.pileup)+'.root'),
    outputCommands = cms.untracked.vstring('keep *_*_*_*',
                                            'drop *_mix_*_*',
                                            'keep *_*_*GPU*_*'
                                            ),
    #outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(options.pileup)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-2)
process.mix.maxBunch = cms.int32(2)


import os
import subprocess
proc = subprocess.Popen(['eos','ls', options.minbias], stdout=subprocess.PIPE)
fList = [os.path.join(options.minbias,x) for x in proc.stdout.read().split() if '.root' in x]

#from random import shuffle
#shuffle(fList)
#process.mix.input.fileNames = cms.untracked.vstring(fList)
process.mix.input.fileNames = cms.untracked.vstring(
            'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_0.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_10.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_100.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1000.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1001.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1002.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1003.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1004.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1005.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1006.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1007.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1008.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1009.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_101.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1010.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1011.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1012.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1013.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1014.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1015.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1016.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1017.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1018.root',
            #'file:/eos/cms/store/cmst3/user/psilva/CMSSW_10_6_0/MinBias/Events_1019.root'
            )



#-------------------
# ntuplizer imports
#-------------------
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_Tree_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_Event_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_GenParticles_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_HGCDigis_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_HBHERecHits_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_HGCRecHits_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_HGCSimHits_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_SimTracks_cfi")
process.load("HGCalAnalysis.HGCalTreeMaker.HGCalTupleMaker_RecoTracks_cfi")

process.hgcalTupleHGCDigisGPU = process.hgcalTupleHGCDigis.clone(  source = cms.untracked.VInputTag(
        cms.untracked.InputTag("mix","HGCDigisEEGPU"),
        cms.untracked.InputTag("mix","HGCDigisHEfrontGPU"),
        cms.untracked.InputTag("simHGCalUnsuppressedDigis","HEback")
        ),
        Prefix = cms.untracked.string  ("GPUHGCDigi")
)

process.hgcalTupleHGCDigisGPUTwo = process.hgcalTupleHGCDigis.clone(  source = cms.untracked.VInputTag(
        cms.untracked.InputTag("mix","HGCDigisEEGPUTwo"),
        cms.untracked.InputTag("mix","HGCDigisHEfrontGPUTwo"),
        cms.untracked.InputTag("simHGCalUnsuppressedDigis","HEback")
        ),
        Prefix = cms.untracked.string  ("GPUTwoHGCDigi")
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Ntu_"+str(options.pileup)+".root")
)
process.ntu = cms.Sequence(
    process.hgcalTupleEvent*
    #process.hgcalTupleGenParticles*
    #process.hgcalTupleHGCSimHits*
    process.hgcalTupleHGCDigis*
    process.hgcalTupleHGCDigisGPU*
    #process.hgcalTupleHGCDigisGPUTwo*
    process.hgcalTupleTree
)
process.ntu_path = cms.Path(
    process.ntu
)



ceeOnGPU=process.mix.digitizers.hgceeDigitizer.clone(digitizationType  = cms.uint32(1),
                                                     digiCollection    = cms.string("HGCDigisEEGPU")
                                                     );
cehOnGPU=process.mix.digitizers.hgchefrontDigitizer.clone(digitizationType  = cms.uint32(1),
                                                          digiCollection    = cms.string("HGCDigisHEfrontGPU")
                                                          );

ceeOnGPUTwo=process.mix.digitizers.hgceeDigitizer.clone(digitizationType  = cms.uint32(1),
                                                     digiCollection    = cms.string("HGCDigisEEGPUTwo")
                                                     );
cehOnGPUTwo=process.mix.digitizers.hgchefrontDigitizer.clone(digitizationType  = cms.uint32(1),
                                                          digiCollection    = cms.string("HGCDigisHEfrontGPUTwo")
                                                          );


process.theDigitizersValid.ceeDigitizerOnGPU =cms.PSet( ceeOnGPU )
process.theDigitizersValid.cehDigitizerOnGPU =cms.PSet( cehOnGPU )
#process.theDigitizersValid.ceeDigitizerOnGPUTwo =cms.PSet( ceeOnGPUTwo )
#process.theDigitizersValid.cehDigitizerOnGPUTwo =cms.PSet( cehOnGPUTwo )

process.mix.digitizers = cms.PSet(process.theDigitizersValid)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.mix)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step, process.ntu_path)
#process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step, process.FEVTDEBUGHLToutput_step])
#from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
#associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
#from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
#process = customizeHLTforMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion


from Validation.Performance.TimeMemoryInfo import customise as customiseTime
process = customiseTime(process)


process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1),
                                     numberOfStreams = cms.untracked.uint32(0))
