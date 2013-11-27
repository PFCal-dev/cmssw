import FWCore.ParameterSet.Config as cms

process = cms.Process("PFCalGeometryTest")
process.load('Geometry.PFCalGeometryBuilder.GeometrySLHCPFCalOnly_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.pfcalGeoAna = cms.EDAnalyzer("PFCalGeometryAnalyzer",
                                     ddViewName     = cms.untracked.string(""),
                                     ddRootNodeName = cms.untracked.string("cms:OCMS"),
                                     runGeometryBuilderFromDDD = cms.untracked.bool(True),
                                     listRoots      = cms.untracked.bool(False),
                                     showSerialized = cms.untracked.bool(False),
                                     geomInfoDumpCfg = cms.PSet( run = cms.untracked.bool(False),
                                                                 dumpGeoHistory = cms.untracked.bool(False),
                                                                 dumpSpecs = cms.untracked.bool(False),
                                                                 dumpPosInfo =cms.untracked.bool(True),
                                                                 outFileName = cms.untracked.string("GeoHistory"),
                                                                 numNodesToDump = cms.untracked.uint32(0)
                                                                 )
                                     )

process.p = cms.Path(process.pfcalGeoAna)

