/**
   \class PFCalNumberingScheme
   Implementation of PFCalNumberingScheme
   \author P. Silva - CERN
 */

#include "Geometry/PFCalGeometry/interface/PFCalNumberingScheme.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"


//
DetId PFCalNumberingScheme::buildDetIdFor()
{

  bool isEE(false);

  DetId newDetId( DetId::Detector::PFCal, isEE ? GeomDetEnumerators::PFCalEE : GeomDetEnumerators::PFCalHE );
  return newDetId;
}
