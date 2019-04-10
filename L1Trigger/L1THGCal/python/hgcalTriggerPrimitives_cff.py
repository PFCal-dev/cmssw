import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFE_cff import *
from L1Trigger.L1THGCal.hgcalConcentrator_cff import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1_cff import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2_cff import *
from L1Trigger.L1THGCal.hgcalTowerMap_cff import *
from L1Trigger.L1THGCal.hgcalTower_cff import *


hgcalTriggerPrimitives = cms.Sequence(hgcalVFE*hgcalConcentrator*hgcalBackEndLayer1*hgcalBackEndLayer2*hgcalTowerMap*hgcalTower)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V9
from L1Trigger.L1THGCal.customCalibration import  custom_cluster_calibration_global, custom_fe_thresholds
modifyHgcalTriggerPrimitivesWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(custom_geometry_V9)
modifyHgcalTriggerPrimitivesCalibWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(lambda process : custom_cluster_calibration_global(process, factor=1))
modifyHgcalTriggerPrimitivesCalibThresholdsWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(lambda process : custom_fe_thresholds(process, fe_threshold=1.5, cl3d_seed_threshold=7.5))

from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
# can't declare a producer version of simHGCalUnsuppressedDigis in the normal flow of things,
# because it's already an EDAlias elsewhere
def _fakeHGCalDigiAlias(process):
	from EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi import HGCDigiConverter as _HGCDigiConverter
	process.simHGCalUnsuppressedDigis = _HGCDigiConverter.clone()
	process.hgcalTriggerPrimitives.insert(0,process.simHGCalUnsuppressedDigis)
doFakeHGCalDigiAlias = convertHGCalDigisSim.makeProcessModifier(_fakeHGCalDigiAlias)
