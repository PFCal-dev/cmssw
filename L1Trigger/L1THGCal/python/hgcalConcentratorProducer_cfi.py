import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

threshold_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                               Method = cms.string('thresholdSelect'),
                               NData = cms.uint32(999),
                               MaxCellsInModule = cms.uint32(288),
                               linLSB = cms.double(100./1024.),
                               adcsaturationBH = adcSaturationBH_MIP,
                               adcnBitsBH = adcNbitsBH,
                               TCThreshold_fC = cms.double(0.),
                               TCThresholdBH_MIP = cms.double(0.),
                               triggercell_threshold_silicon = cms.double(2.), # MipT
                               triggercell_threshold_scintillator = cms.double(2.), # MipT
                               coarsenTriggerCells = cms.bool(False)
                               )


best_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.string('bestChoiceSelect'),
                          NData = cms.uint32(12),
                          MaxCellsInModule = cms.uint32(288),
                          linLSB = cms.double(100./1024.),
                          adcsaturationBH = adcSaturationBH_MIP,
                          adcnBitsBH = adcNbitsBH,
                          TCThreshold_fC = cms.double(0.),
                          TCThresholdBH_MIP = cms.double(0.),
                          triggercell_threshold_silicon = cms.double(0.),
                          triggercell_threshold_scintillator = cms.double(0.),
                          coarsenTriggerCells = cms.bool(False)
                          )


supertc_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('superTriggerCell'),# superTriggerCell,oneBitFraction,equalShare
                             stcSize = cms.vuint32(4,4,4),
                             fixedDataSizePerHGCROC = cms.bool(False),
                             coarsenTriggerCells = cms.bool(False)
                             )

typeenergydivision_onebitfraction_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('oneBitFraction'),
                             stcSize = cms.vuint32(4,8,8),
                             fixedDataSizePerHGCROC = cms.bool(True),
                             coarsenTriggerCells = cms.bool(False),
                             oneBitFractionThreshold = 0.125,
                             oneBitFractionLowValue = 0.0625,
                             oneBitFractionHighValue = 0.25,
                             )


typeenergydivision_equalshare_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('equalShare'),
                             stcSize = cms.vuint32(4,8,8),
                             fixedDataSizePerHGCROC = cms.bool(True),
                             coarsenTriggerCells = cms.bool(False),
                             nTriggerCellsForDivision = 4
)





from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
# V9 samples have a different defintiion of the dEdx calibrations. To account for it
# we reascale the thresholds of the FE selection
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV9.toModify(threshold_conc_proc,
                        triggercell_threshold_silicon=cms.double(1.5),  # MipT
                        triggercell_threshold_scintillator=cms.double(1.5),  # MipT
                        )

hgcalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = threshold_conc_proc.clone()
    )
