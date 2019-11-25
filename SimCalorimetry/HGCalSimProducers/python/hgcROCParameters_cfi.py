import FWCore.ParameterSet.Config as cms

#define base parameters for the HGCROC simulation in the SimCalorimetry/HGCalSimProducers/src/HGCFEElectronics.cc

hgcROCSettings = cms.PSet( 
    # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
    fwVersion         = cms.uint32(2),
    # leakage to bunches -2, -1, in-time, +1, +2, +3 (from J. Kaplon)
    #NOTE: this is a fixed-size array inside the simulation (for speed) change accordingly!
    adcPulse          = cms.vdouble(0.00, 0.017,   0.817,   0.163,  0.003,  0.000),
    pulseAvgT         = cms.vdouble(0.00, 23.42298,13.16733,6.41062,5.03946,4.5320),
    # n bits for the ADC
    adcNbits          = cms.uint32(10),
    # ADC saturation
    adcSaturation_fC  = cms.double(100),
    # the tdc resolution smearing (in picoseconds)
    tdcResolutionInPs = cms.double( 0.001 ),
    # jitter for timing noise term ns
    jitterNoise_ns = cms.vdouble(25., 25., 25.),
    # jitter for timing noise term ns
    jitterConstant_ns = cms.vdouble(0.0004, 0.0004, 0.0004),
    # LSB for TDC, assuming 12 bit dynamic range to 10 pC
    tdcNbits          = cms.uint32(12),
    # TDC saturation
    tdcSaturation_fC  = cms.double(10000),
    # aim to have the MIP peak at 10 ADC
    targetMIPvalue_ADC   = cms.uint32(10),
    # raise threshold flag (~MIP/2) this is scaled
    # for different thickness
    adcThreshold_fC   = cms.double(0.672),
    # raise usage of TDC and mode flag (from J. Kaplon)
    tdcOnset_fC       = cms.double(60),
    # raise usage of TDC for TOA only
    tdcForToAOnset_fC = cms.vdouble(12., 12., 12.),
    # LSB for time of arrival estimate from TDC in ns
    toaLSB_ns         = cms.double(0.0244),
    #toa computation mode (0=by weighted energy, 1=simple threshold)
    toaMode           = cms.uint32(1),
    # TDC charge drain parameterisation (from J. Kaplon)
    tdcChargeDrainParameterisation = cms.vdouble(
        -919.13, 365.36, -14.10, 0.2,
        -21.85, 49.39,  22.21,  0.8,
        -0.28,   27.14,  43.95,
        3.89048 )
)

hgcROCSettingsForHEback = hgcROCSettings.clone( adcNbits        = cms.uint32(10),      #two bits less (uniform with the silicon)
                                                adcSaturation_fC = cms.double(68.75),  #one quarter value of pre12
                                                tdcSaturation_fC  = cms.double(1000),  #allow up to 1000 MIPs
                                                targetMIPvalue_ADC   = cms.uint32(15), #to be used for HGCROC gain proposal
                                                adcThreshold_fC = cms.double(0.5),     #unchanged with respect to pre12
                                                tdcOnset_fC       = cms.double(55),    #turn TDC when 80% of the ADC range is reached (one quarter of pre12
                                                #                                       indicative at this point)
                                                tdcForToAOnset_fC = cms.vdouble(12.,12.,12.),  #turn ToA for 20% of the TDC threshold (indicative at this point)
                                            )
