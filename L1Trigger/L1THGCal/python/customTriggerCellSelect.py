import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc, coarsetc_onebitfraction_proc, coarsetc_equalshare_proc

def custom_triggercellselect_supertriggercell(process,
                                              stcSize=supertc_conc_proc.stcSize,
                                              type_energy_division=supertc_conc_proc.type_energy_division,
                                              fixedDataSizePerHGCROC=supertc_conc_proc.fixedDataSizePerHGCROC
                                              ):
    parameters = supertc_conc_proc.clone(stcSize = stcSize,
                                         type_energy_division = type_energy_division,
                                         fixedDataSizePerHGCROC = fixedDataSizePerHGCROC  
                                         )
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_threshold(process,
                                       threshold_silicon=threshold_conc_proc.threshold_silicon,  # in mipT
                                       threshold_scintillator=threshold_conc_proc.threshold_scintillator,  # in mipT
                                       coarsenTriggerCells=threshold_conc_proc.coarsenTriggerCells  
                                       ):
    parameters = threshold_conc_proc.clone(
            threshold_silicon = threshold_silicon,
            threshold_scintillator = threshold_scintillator,
            coarsenTriggerCells = coarsenTriggerCells  
            )
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_bestchoice(process,
                                        triggercells=best_conc_proc.NData
                                        ):
    parameters = best_conc_proc.clone(NData = triggercells)
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_coarsetc_onebitfraction(process,
                                   oneBitFractionThreshold = coarsetc_onebitfraction_proc.oneBitFractionThreshold,
                                   oneBitFractionLowValue = coarsetc_onebitfraction_proc.oneBitFractionLowValue,
                                   oneBitFractionHighValue = coarsetc_onebitfraction_proc.oneBitFractionHighValue,
                                   ):
    parameters = coarsetc_onebitfraction_proc.clone(
        oneBitFractionThreshold = oneBitFractionThreshold,
        oneBitFractionLowValue = oneBitFractionLowValue,
        oneBitFractionHighValue = oneBitFractionHighValue,
    )

    return process




def custom_coarsetc_equalshare(process,
                               ):
    parameters = coarsetc_equalshare_proc.clone()

    return process
    
