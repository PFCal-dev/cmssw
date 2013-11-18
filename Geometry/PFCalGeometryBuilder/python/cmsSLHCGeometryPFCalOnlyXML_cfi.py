import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                                                                   'Geometry/PFCalGeometryBuilder/data/PFCAL_geometry_v1.xml'),
                                        rootNodeName = cms.string('cmsPartFlowCalorimeter:PFCAL')
                                        )

