/**
   \class PFCalNumberingScheme
   Implementation of PFCalNumberingScheme
   \author P. Silva - CERN
 */

#include "Geometry/PFCalGeometry/interface/PFCalNumberingScheme.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include <iostream>


//
DetId PFCalNumberingScheme::buildDetIdFor(bool isPos,bool isHE, unsigned int layerId, unsigned int sectorId)
{

  int posOffset=DetId::kSubdetOffset-1;
  int layerOffset=posOffset-8;
  int sectorOffset=posOffset-8;

  uint32_t newDetId(0);
  newDetId          |= ((DetId::Detector::PFCal & 0xf)                   << DetId::kDetOffset);
  if(isHE) newDetId |= ((GeomDetEnumerators::SubDetector::PFCalHE & 0xf) << DetId::kSubdetOffset);
  else     newDetId |= ((GeomDetEnumerators::SubDetector::PFCalEE & 0xf) << DetId::kSubdetOffset);
  newDetId          |= (isPos                                            << posOffset);
  newDetId          |= ((layerId & 0xff)                                 << layerOffset );
  newDetId          |= ((sectorId & 0xff)                                << sectorOffset );

  return DetId(newDetId);
}
